import argparse

from data_provider import datasets_factory


parser = argparse.ArgumentParser("please give appropriate arguments")
parser.add_argument('--mode', default='train', type=str)
parser.add_argument('--model_name', default='e3d_lstm', type=str)
parser.add_argument('--dataset_name', default='mnist', type=str)  #action
parser.add_argument('--dir_test_result',default='test_result', type=str)
parser.add_argument('--train_data_paths', default=None, type=str)
parser.add_argument('--valid_data_paths', default=None, type=str)
parser.add_argument('--n_gpu', default=1, type=int)
parser.add_argument('--interval_print', default=10, type=int)
parser.add_argument('--use_cuda', default=True, type=bool)
parser.add_argument('--epoch', default=10, type=int)
parser.add_argument('--batch_size', default=2, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--dropout', default=0.5, type=float)
parser.add_argument('--num_hidden', default='64, 64, 64, 64', type=str)
parser.add_argument('--seq_length', default=5, type=int)
parser.add_argument('--input_length', default=10, type=int)
parser.add_argument('--total_length', default=40, type=int)
parser.add_argument('--img_width', default=64, type=int)
parser.add_argument('--img_height', default=64, type=int)
parser.add_argument('--patch_size', default=4, type=int)
parser.add_argument('--img_channel', default=1, type=int)
parser.add_argument('--scheduled_sampling', default=True, type=bool)
parser.add_argument('--sampling_stop_iter', default=100000, type=int)
parser.add_argument('--sampling_changing_rate', default=0.00001, type=float)
parser.add_argument('--sampling_start_value', default=1.0, type=float)
parser.add_argument('--max_iterations', default=200000, type=int)

args = parser.parse_args()

# setattr(args, 'train_data_paths', "../kth_action")
# setattr(args, 'valid_data_paths', "../bai/kth_action")
setattr(args, 'train_data_paths', "../moving_mnist_example/moving-mnist-train.npz")
setattr(args, 'valid_data_paths', "../moving_mnist_example/moving-mnist-valid.npz")
# load data
train_input_handle, test_input_handle = datasets_factory.data_provider(
args.dataset_name,
args.train_data_paths,
args.valid_data_paths,
args.batch_size,
args.img_width,
seq_length=args.total_length,
is_training=True)

eta = args.sampling_start_value


for itr in range(1, 8):
    if train_input_handle.no_batch_left():
        train_input_handle.begin(do_shuffle=True)
    ims = train_input_handle.get_batch()
    # ims = preprocess.reshape_patch(ims, args.patch_size)
    print("ims.shape:{}".format(ims.shape))