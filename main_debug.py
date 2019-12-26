import paddle.fluid as fluid
import numpy as np

from models.lstm_cell import  ConvLSTMCell

batch_size = 16
timesteps = 20
shape = [30, 40, 50]
kernel = [3, 3, 3]
channels = 2
filters = 8

# Create a placeholder for videos.
inputs = fluid.data(name='input', shape=[batch_size]+shape+[channels])
labels = fluid.data(name='label', shape=[batch_size]+shape+[filters])

# Add the ConvLSTM step.
cell = ConvLSTMCell(shape, filters, kernel)
# outputs, state = tf.nn.dynamic_rnn(cell, inputs, dtype=inputs.dtype)
c0 = fluid.layers.zeros(shape=[batch_size]+shape+[filters], dtype='float32')
h0 = fluid.layers.zeros(shape=[batch_size]+shape+[filters], dtype='float32')
s0 = [h0, c0]
h, c = cell(inputs, s0)
# print(h.shape)

place = fluid.CPUPlace()
exe = fluid.Executor(place)

x = np.random.random([batch_size]+shape+[channels]).astype( dtype='float32')
y = np.random.random([batch_size]+shape+[filters]).astype( dtype='float32')

mse = fluid.layers.mse_loss(h, labels)
optimizer = fluid.optimizer.Adam(learning_rate=0.001)
optimizer.minimize(mse)

exe.run(fluid.default_startup_program())
for i in range(1000):
    rs = exe.run(feed={'input':x, 'label':y}, fetch_list=[mse])
    if i%10 == 0:
        print(i, rs[0])


