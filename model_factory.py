# Copyright 2019 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Factory to get E3D-LSTM models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import paddle.fluid as fluid

from models import eidetic_3d_lstm_net


class Model(object):
    """Model class for E3D-LSTM model."""

    def __init__(self, configs):
        self.configs = configs
        self.x = fluid.data(name='x', dtype='float32', shape=[
            self.configs.batch_size, self.configs.total_length,
            self.configs.img_width // self.configs.patch_size,
            self.configs.img_width // self.configs.patch_size,
            self.configs.patch_size * self.configs.patch_size *
            self.configs.img_channel
        ])
        self.real_input_flag = fluid.data(name='real_input_flag', dtype='float32', shape=[
            self.configs.batch_size,
            self.configs.total_length - self.configs.input_length - 1,
            self.configs.img_width // self.configs.patch_size,
            self.configs.img_width // self.configs.patch_size,
            self.configs.patch_size * self.configs.patch_size *
            self.configs.img_channel
        ])
        self.num_hidden = [int(x) for x in self.configs.num_hidden.split(',')]
        self.num_layers = len(self.num_hidden)
        print('self.num_hidden:{}'.format(self.num_hidden))
        print('self.num_layers:{}'.format(self.num_layers))
        self.startup_program = fluid.default_startup_program()
        self.main_program = fluid.default_main_program()

        self.output_list = self.construct_model(self.x, self.real_input_flag,
                                                self.num_layers, self.num_hidden)

        self.gen_imgs = self.output_list[0]
        self.ave_loss = self.output_list[1]
        # self.test_program = self.main_program.clone(for_test=True)
        self.optimizer = fluid.optimizer.Adam(learning_rate=configs.lr)
        self.optimizer.minimize(self.ave_loss)

        # loss_train.append(loss / self.configs.batch_size)
        # # gradients
        # all_params = tf.trainable_variables()
        # grads.append(tf.gradients(loss, all_params))
        # self.pred_seq.append(gen_ims)

    def train(self, inputs, real_input_flag, exe, place):

        # feeder = fluid.DataFeeder(feed_list=[self.x, self.real_input_flag], place=place)
        exe.run(self.startup_program)
        gen_imgs, ave_loss = exe.run(self.main_program, feed={'x':inputs, 'real_input_flag':real_input_flag},
                      fetch_list=[self.gen_imgs, self.ave_loss])
        return gen_imgs, ave_loss

    def test(self, inputs, real_input_flag, exe, place):

        pass

    def save(self, itr):
        pass

    def load(self, checkpoint_path):
        pass

    def construct_model(self, images, real_input_flag, num_layers, num_hidden):
        """Contructs a model."""
        networks_map = {
            'e3d_lstm': eidetic_3d_lstm_net.rnn,
        }

        if self.configs.model_name in networks_map:
            func = networks_map[self.configs.model_name]
            # return [None, None]
            return func(images, real_input_flag, num_layers, num_hidden, self.configs)
        else:
            raise ValueError('Name of network unknown %s' % self.configs.model_name)
