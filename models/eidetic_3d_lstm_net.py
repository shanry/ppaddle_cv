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
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

# from src.layers.rnn_cell import Eidetic3DLSTMCell as eidetic_lstm
# import tensorflow as tf
import numpy as np



def rnn(images, real_input_flag, num_layers, num_hidden, configs):
  """Builds a RNN according to the config."""
  gen_images, lstm_layer, cell, hidden, c_history = [], [], [], [], []
  shape = images.get_shape().as_list()
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
    if i == 0:
      num_hidden_in = output_channels
    else:
      num_hidden_in = num_hidden[i - 1]
    new_lstm = None
    # new_lstm = eidetic_lstm(
    #     name='e3d' + str(i),
    #     input_shape=[ims_width, window_length, ims_height, num_hidden_in],
    #     output_channels=num_hidden[i],
    #     kernel_shape=[2, 5, 5])
    lstm_layer.append(new_lstm)
    zero_state = np.zeros(
        [batch_size, window_length, ims_width, ims_height, num_hidden[i]])
    cell.append(zero_state)
    hidden.append(zero_state)
    c_history.append(None)

  memory = zero_state



  return [None, None]
