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
        print("xhc.shape:{}".format(xhc.shape))
        conv_xhc = fluid.layers.conv3d(xhc, num_filters=self._filters*2, filter_size=self._kernel,
                                       padding='same', data_format=self._data_format)
        print("conv_xhc.shape:{}".format(conv_xhc))
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

        c = f*c + i*c_temp

        xhc_2 = fluid.layers.concat([x, h, c], axis=-1)
        o = fluid.layers.conv3d(xhc_2, self._filters, self._kernel,
                                padding='same', data_format=self._data_format)
        if self._normalize:
            o = fluid.layers.layer_norm(o)
            c = fluid.layers.layer_norm(c)
        o = fluid.layers.sigmoid(o)

        h = o*fluid.layers.tanh(c)

        return h, (h, c)
