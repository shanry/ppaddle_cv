import paddle.fluid as fluid


class ConvLSTMCell():
    """A LSTM cell with convolutions instead of multiplications.

    Reference:
    Xingjian, S. H. I., et al. "Convolutional LSTM network: A machine learning approach for precipitation nowcasting." Advances in Neural Information Processing Systems. 2015.
    """

    def __init__(self, shape, filters, kernel, normalize=True, data_format='NDHWC'):
        self._shape = shape
        self._kernel = kernel
        self._filters = filters
        self._normalize = normalize
        self._data_format = data_format
        if data_format == 'NDHWC':
            self._size = shape + [self._filters]
            self._feature_axis = -1
        elif data_format == 'NCDHW':
            self._size = [self._filters] + shape
            self._feature_axis = 0
        else:
            raise ValueError('Unknown data_format')

    def __call__(self, x, state):
        h, c = state
        xhc = fluid.layers.concat([x, h, c], axis=-1)
        # print("xhc.shape:{}".format(xhc.shape))
        conv_xhc = fluid.layers.conv3d(xhc, num_filters=self._filters * 2, filter_size=self._kernel,
                                       padding='same', data_format=self._data_format)
        # print("conv_xhc.shape:{}".format(conv_xhc))
        i, f = fluid.layers.split(conv_xhc, 2)
        xh = fluid.layers.concat([x, h], axis=-1)
        c_temp = fluid.layers.conv3d(xh, self._filters, self._kernel,
                                     padding='same', data_format=self._data_format)
        if self._normalize:
            i = fluid.layers.layer_norm(i)
            f = fluid.layers.layer_norm(f)
            c_temp = fluid.layers.layer_norm(c_temp)
        i = fluid.layers.sigmoid(i)
        f = fluid.layers.sigmoid(f)
        c_temp = fluid.layers.tanh(c_temp)

        # print("f.shape{}".format(f.shape))
        # print("i.shape{}".format(i.shape))
        # print("c.shape{}".format(c.shape))
        # print("c_temp.shape{}".format(c_temp.shape))

        c = f * c + i * c_temp

        xhc_2 = fluid.layers.concat([x, h, c], axis=-1)
        o = fluid.layers.conv3d(xhc_2, self._filters, self._kernel,
                                padding='same', data_format=self._data_format)
        if self._normalize:
            o = fluid.layers.layer_norm(o)
            c = fluid.layers.layer_norm(c)
        o = fluid.layers.sigmoid(o)
        # print("o.shape{}".format(o.shape))
        # print("c.shape{}".format(o.shape))
        h = o * fluid.layers.tanh(c)

        return h, c


class EideticLSTMCell():
    """A LSTM cell with convolutions instead of multiplications.

    Reference:
    Xingjian, S. H. I., et al. "Convolutional LSTM network: A machine learning approach for precipitation nowcasting." Advances in Neural Information Processing Systems. 2015.
    """

    def __init__(self, shape, filters, kernel, name, normalize=True, data_format='NDHWC'):
        self._shape = shape
        self._filters = filters
        self._kernel = kernel
        self._normalize = normalize
        self._data_format = data_format
        self._name = name
        if data_format == 'NDHWC':
            self._size = shape + [self._filters]
            self._feature_axis = -1
        elif data_format == 'NCDHW':
            self._size = [self._filters] + shape
            self._feature_axis = 0
        else:
            raise ValueError('Unknown data_format')

    def _attn(self, in_query, in_keys, in_values):
        """3D Self-Attention Block.

        Args:
          in_query: Tensor of shape (b,l,w,h,n).
          in_keys: Tensor of shape (b,attn_length,w,h,n).
          in_values: Tensor of shape (b,attn_length,w,h,n).

        Returns:
          attn: Tensor of shape (b,l,w,h,n).

        Raises:
          ValueError: If any number of dimensions regarding the inputs is not 4 or 5
            or if the corresponding dimension lengths of the inputs are not
            compatible.
        """
        q_shape = list(in_query.shape)
        if len(q_shape) == 4:
            batch = q_shape[0]
            width = q_shape[1]
            height = q_shape[2]
            num_channels = q_shape[3]
        elif len(q_shape) == 5:
            batch = q_shape[0]
            width = q_shape[2]
            height = q_shape[3]
            num_channels = q_shape[4]
        else:
            raise ValueError("Invalid input_shape {} for the query".format(q_shape))

        k_shape = list(in_keys.shape)
        if len(k_shape) != 5:
            raise ValueError("Invalid input_shape {} for the keys".format(k_shape))

        v_shape = list(in_values.shape)
        if len(v_shape) != 5:
            raise ValueError("Invalid input_shape {} for the values".format(v_shape))

        if width != k_shape[2] or height != k_shape[3] or num_channels != k_shape[4]:
            raise ValueError("Invalid input_shape {} and {}, not compatible.".format(
                q_shape, k_shape))
        if width != v_shape[2] or height != v_shape[3] or num_channels != v_shape[4]:
            raise ValueError("Invalid input_shape {} and {}, not compatible.".format(
                q_shape, v_shape))
        if k_shape[2] != v_shape[2] or k_shape[3] != v_shape[3] or k_shape[
            4] != v_shape[4]:
            raise ValueError("Invalid input_shape {} and {}, not compatible.".format(
                k_shape, v_shape))

        query = fluid.layers.reshape(in_query, [batch, -1, num_channels])
        keys = fluid.layers.reshape(in_keys, [batch, -1, num_channels])
        values = fluid.layers.reshape(in_values, [batch, -1, num_channels])
        attn = fluid.layers.matmul(query, keys, False, True)
        attn = fluid.layers.softmax(attn, axis=2)
        attn = fluid.layers.matmul(attn, values, False, False)
        if len(q_shape) == 4:
            attn = fluid.layers.reshape(attn, [batch, width, height, num_channels])
        else:
            attn = fluid.layers.reshape(attn, [batch, -1, width, height, num_channels])
        return attn

    def __call__(self, inputs, hidden, cell, global_memory, eidetic_cell):

        # print("this is EideticLSTMCell")

        # print("inputs.shape:{}".format(inputs.shape))
        # print("hidden.shape:{}".format(hidden.shape))
        # print("cell.shape:{}".format(cell.shape))
        # print("global_memory.shape:{}".format(global_memory.shape))
        #
        # print("******************************************")

        # with tf.variable_scope(self._layer_name):
        new_hidden = fluid.layers.conv3d(hidden, 4 * self._filters,
                                         self._kernel, padding='same',
                                         data_format=self._data_format,
                                         param_attr=fluid.param_attr.ParamAttr(
                                             name=self._name + "_conv3d" + "_new_hidden"),
                                         bias_attr=fluid.param_attr.ParamAttr(
                                             name=self._name + "_conv3d" + "_new_hidden_b")
                                         )
        if self._normalize:
            new_hidden = fluid.layers.layer_norm(new_hidden, param_attr=fluid.param_attr.ParamAttr(
                name=self._name + "_layer_norm" + "_new_hidden"), bias_attr=fluid.param_attr.ParamAttr(
                name=self._name + "_layer_norm" + "_new_hidden_b"))
        i_h, g_h, r_h, o_h = fluid.layers.split(new_hidden, 4)
        new_inputs = fluid.layers.conv3d(inputs, 7 * self._filters, self._kernel,
                                         padding='same', data_format=self._data_format,
                                         param_attr=fluid.param_attr.ParamAttr(
                                             name=self._name + "_conv3d" + "_new_inputs"),
                                         bias_attr=fluid.param_attr.ParamAttr(
                                             name=self._name + "_conv3d" + "_new_inputs_b")
                                         )
        if self._normalize:
            new_inputs = fluid.layers.layer_norm(new_inputs, param_attr=fluid.param_attr.ParamAttr(
                name=self._name + "_layer_norm" + "_new_inputs"), bias_attr=fluid.param_attr.ParamAttr(
                name=self._name + "_layer_norm" + "_new_inputs_b"))
        i_x, g_x, r_x, o_x, temp_i_x, temp_g_x, temp_f_x = fluid.layers.split(
            new_inputs, 7)

        i_t = fluid.layers.sigmoid(i_x + i_h)
        r_t = fluid.layers.sigmoid(r_x + r_h)
        g_t = fluid.layers.tanh(g_x + g_h)

        # print("new_hidden.shape:{}".format(new_hidden.shape))
        # print("new_inputs.shape:{}".format(new_inputs.shape))

        kv = fluid.layers.concat(eidetic_cell, axis=1)
        new_cell = cell + self._attn(r_t, kv, kv)
        new_cell = fluid.layers.layer_norm(new_cell, param_attr=fluid.param_attr.ParamAttr(
            name=self._name + "_layer_norm" + "_new_cell"), bias_attr=fluid.param_attr.ParamAttr(
            name=self._name + "_layer_norm" + "_new_cell_b")) + i_t * g_t

        new_global_memory = fluid.layers.conv3d(global_memory, 4 * self._filters,
                                                self._kernel, padding='same', data_format=self._data_format,
                                                param_attr=fluid.param_attr.ParamAttr(
                                                    name=self._name + "_conv3d" + "_new_global_memory"),
                                                bias_attr=fluid.param_attr.ParamAttr(
                                                    name=self._name + "_conv3d" + "_new_global_memory_b")
                                                )

        if self._normalize:
            new_global_memory = fluid.layers.layer_norm(new_global_memory, param_attr=fluid.param_attr.ParamAttr(
                name=self._name + "_layer_norm" + "_new_global_memory"), bias_attr=fluid.param_attr.ParamAttr(
                name=self._name + "_layer_norm" + "_new_global_memory_b"))
            i_m, f_m, g_m, m_m = fluid.layers.split(new_global_memory, 4)

        temp_i_t = fluid.layers.sigmoid(temp_i_x + i_m)
        temp_f_t = fluid.layers.sigmoid(temp_f_x + f_m)  # original: + self._forget_bias
        temp_g_t = fluid.layers.tanh(temp_g_x + g_m)
        new_global_memory = temp_f_t * fluid.layers.tanh(m_m) + temp_i_t * temp_g_t

        # print("temp_i_t.shape:{}".format(temp_i_t.shape))
        # print("temp_f_t.shape:{}".format(temp_f_t.shape))
        # print("temp_g_t.shape:{}".format(temp_g_t.shape))
        # print("new_global_memory.shape:{}".format(new_global_memory.shape))

        o_c = fluid.layers.conv3d(new_cell, self._filters,
                                  self._kernel, padding='same', data_format=self._data_format,
                                  param_attr=fluid.param_attr.ParamAttr(
                                      name=self._name + "_conv3d" + "_o_c"),
                                  bias_attr=fluid.param_attr.ParamAttr(
                                      name=self._name + "_conv3d" + "_o_c_b")
                                  )
        o_m = fluid.layers.conv3d(new_global_memory, self._filters, self._kernel,
                                  padding='same', data_format=self._data_format,
                                  param_attr=fluid.param_attr.ParamAttr(
                                      name=self._name + "_conv3d" + "_o_m"),
                                  bias_attr=fluid.param_attr.ParamAttr(
                                      name=self._name + "_conv3d" + "_o_m_b")
                                  )

        output_gate = fluid.layers.tanh(o_x + o_h + o_c + o_m)

        # print("new_cell.shape:{}".format(new_cell.shape))
        memory = fluid.layers.concat([new_cell, new_global_memory], -1)
        memory = fluid.layers.conv3d(memory, self._filters, 1,
                                     padding='same', data_format=self._data_format,
                                     param_attr=fluid.param_attr.ParamAttr(
                                         name=self._name + "_conv3d" + "_memory"),
                                     bias_attr=fluid.param_attr.ParamAttr(
                                         name=self._name + "_conv3d" + "_memory_b")
                                     )

        output = fluid.layers.tanh(memory) * fluid.layers.sigmoid(output_gate)
        # print("output.shape:{}".format(output.shape))
        # print("*****************************************")
        return output, new_cell, global_memory
