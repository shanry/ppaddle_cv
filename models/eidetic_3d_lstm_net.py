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

"""Builds an E3D RNN."""
import numpy as np
import paddle.fluid as fluid

from models.lstm_cell import ConvLSTMCell as conv_lstm
from models.lstm_cell import EideticLSTMCell as eide_lstm


def rnn(images, real_input_flag, num_layers, num_hidden, configs):
    """Builds a RNN according to the config."""
    gen_images, lstm_layer, cell, hidden, c_history, hidden_0, cell_0 = [], [], [], [], [], [], []
    shape = list(images.shape)
    batch_size = shape[0]
    # seq_length = shape[1]
    ims_width = shape[2]
    ims_height = shape[3]
    output_channels = shape[-1]
    # filter_size = configs.filter_size
    total_length = configs.total_length
    input_length = configs.input_length

    window_length = 2
    window_stride = 1

    for i in range(num_layers):
        print("num_layers:{}".format(i))
        if i == 0:
            num_hidden_in = output_channels
        else:
            num_hidden_in = num_hidden[i - 1]
        print("num_hidden_in:{}".format(num_hidden_in))
        if configs.lstm == 'conv':
            new_lstm = conv_lstm(
                shape=[window_length, ims_height, ims_width, num_hidden_in],
                filters=num_hidden[i],
                kernel=[2, 5, 5])
        else:
            new_lstm = eide_lstm(
                shape=[window_length, ims_height, ims_width, num_hidden_in],
                filters=num_hidden[i],
                kernel=[2, 5, 5])
        lstm_layer.append(new_lstm)
        zero_h = fluid.layers.zeros(
            [batch_size, window_length, ims_width, ims_height, num_hidden[i]], dtype='float32')
        zero_c = fluid.layers.zeros(
            [batch_size, window_length, ims_width, ims_height, num_hidden[i]], dtype='float32')
        hidden_0.append(zero_h)
        cell_0.append(zero_c)
        cell.append([])
        hidden.append([])
        c_history.append(None)

    memory = fluid.layers.zeros(
            [batch_size, window_length, ims_width, ims_height, num_hidden[i]], dtype='float32')
    input_list = []
    for time_step in range(window_length - 1):
        # input_list.append(
        #     fluid.layers.zeros([batch_size, ims_height, ims_width, output_channels], dtype='float32'))

        input_list.append(images[:,1])

    for time_step in range(total_length - 1):
        # with tf.variable_scope('e3d-lstm', reuse=reuse):
        # input_frm = images[:, time_step]
        if time_step < input_length:
            input_frm = images[:, time_step]
        else:
            time_diff = time_step - input_length
            input_frm = real_input_flag[:, time_diff] * images[:, time_step] \
                        + (1 - real_input_flag[:, time_diff]) * x_gen  # pylint: disable=used-before-assignment
        input_list.append(input_frm)

        if time_step % (window_length - window_stride) == 0:
            input_frm = fluid.layers.stack(input_list[time_step:])
            input_frm = fluid.layers.transpose(input_frm, [1, 0, 2, 3, 4])

            for i in range(num_layers):
                # if time_step == 0:
                #     c_history[i] = cell[i]
                # else:
                #     c_history[i] = fluid.layers.concat([c_history[i], cell[i]], 1)
                if i == 0:
                    inputs = input_frm
                else:
                    # inputs = input_frm
                    inputs = hidden[i - 1][-1]
                if time_step == 0:
                    h = fluid.layers.zeros([batch_size, window_length, ims_width,
                                            ims_height, num_hidden[i]], dtype='float32')
                    c = fluid.layers.zeros([batch_size, window_length, ims_width,
                                            ims_height, num_hidden[i]], dtype='float32')
                else:
                    # print('timestep:{}'.format(time_step))
                    h = hidden[i][-1]
                    c = cell[i][-1]
                if configs.lstm == 'conv':
                    h_new, c_new = lstm_layer[i](inputs, (h, c))
                else:
                    if time_step == 0:
                        history = [fluid.layers.zeros([batch_size, window_length, ims_width,
                                            ims_height, num_hidden[i]], dtype='float32')]
                        h_new, c_new, memory = lstm_layer[i](inputs, h, c, memory, history)
                    else:
                        h_new, c_new, memory = lstm_layer[i](inputs, h, c, memory, cell[i])
                hidden[i].append(h_new)
                cell[i].append(c_new)

            x_gen = fluid.layers.conv3d(input=hidden[num_layers - 1][-1], num_filters=output_channels,
                                        filter_size=[window_length, 1, 1], stride=[window_length, 1, 1],
                                        padding='same', data_format='NDHWC')
            x_gen = fluid.layers.squeeze(x_gen, axes=[1])
            # print("hidden[num_layers - 1][-1][:,1].shape:{}".format(hidden[num_layers - 1][-1][:,1].shape))
            # x_gen = hidden[num_layers - 1][-1][:,1]
            # x_gen = fluid.layers.squeeze(x_gen, axes=[1])
            gen_images.append(x_gen)
    print("len(hidden):{}".format(len(hidden)))
    print("len(hidden[0]):{}".format(len(hidden[0])))
    print("len(cell):{}".format(len(cell)))
    print("len(cell[0]):{}".format(len(cell[0])))
    print("len(gen_images):{}".format(len(gen_images)))
    gen_images = fluid.layers.stack(gen_images)
    gen_images = fluid.layers.transpose(gen_images, [1, 0, 2, 3, 4])
    print("gen_images.shape:{}".format(gen_images.shape))
    print("images[:, 1:].shape:{}".format(images[:, 1:].shape))

    loss_l2 = fluid.layers.square_error_cost(gen_images, images[:, 1:])
    loss = fluid.layers.sum(loss_l2)
    # loss = fluid.layers.mse_loss(gen_images, images[:, 1:])
    loss_l1 = fluid.layers.reduce_sum(fluid.layers.abs(gen_images - images[:, 1:]))
    loss += loss_l1
    loss /= batch_size
    out_len = total_length - input_length
    out_ims = gen_images[:, -out_len:]

    return out_ims, loss


def rnn_2(images, real_input_flag, num_layers, num_hidden, configs):
    """Builds a RNN according to the config."""
    gen_images, lstm_layer, cell, hidden, c_history, hidden_0, cell_0 = [], [], [], [], [], [], []
    shape = list(images.shape)
    batch_size = shape[0]
    # seq_length = shape[1]
    ims_width = shape[2]
    ims_height = shape[3]
    output_channels = shape[-1]
    # filter_size = configs.filter_size
    total_length = configs.total_length
    input_length = configs.input_length
    window_length = 5
    cell = conv_lstm(
            shape=[window_length, ims_height, ims_width, output_channels],
            filters=output_channels,
            kernel=[2, 5, 5])
    # outputs, state = tf.nn.dynamic_rnn(cell, inputs, dtype=inputs.dtype)
    c0 = fluid.layers.zeros(shape=[batch_size,window_length, ims_height, ims_width, output_channels], dtype='float32')
    h0 = fluid.layers.zeros(shape=[batch_size,window_length, ims_height, ims_width, output_channels], dtype='float32')
    s0 = [h0, c0]
    inputs = images[:, :window_length]
    h, c = cell(inputs, s0)

    los = fluid.layers.mse_loss(h[:,1],images[:,window_length])

    return h, los

