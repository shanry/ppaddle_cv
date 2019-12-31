import argparse
import datetime
import os
import shutil

import cv2
import numpy as np
import paddle.fluid as fluid

from data_provider import datasets_factory
from model_factory import Model
from utils import preprocess


def batch_psnr(gen_frames, gt_frames):
  """Computes PSNR for a batch of data."""
  if gen_frames.ndim == 3:
    axis = (1, 2)
  elif gen_frames.ndim == 4:
    axis = (1, 2, 3)
  x = np.int32(gen_frames)
  y = np.int32(gt_frames)
  num_pixels = float(np.size(gen_frames[0]))
  mse = np.sum((x - y)**2, axis=axis, dtype=np.float32) / num_pixels
  psnr = 20 * np.log10(255) - 10 * np.log10(mse)
  return np.mean(psnr)


def train(args, model):
    if args.train_data_paths is None or args.valid_data_paths is None:
        raise ValueError("the data paths mush be given !!")
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

    optimizer = fluid.optimizer.Adam(args.lr, 0.95, 0.995)
    optimizer.minimize(model.ave_loss)

    place = fluid.CUDAPlace(0) if args.use_cuda==1 else fluid.CPUPlace()
    print("place:", place)
    exe = fluid.Executor(place)
    start_program = fluid.default_startup_program()
    main_program = fluid.default_main_program()
    exe.run(start_program)


    for itr in range(1, args.max_iterations + 1):
        if train_input_handle.no_batch_left():
            train_input_handle.begin(do_shuffle=True)
        ims = train_input_handle.get_batch()
        ims = preprocess.reshape_patch(ims, args.patch_size)
        # print("ims.shape:{}".format(ims.shape))

        eta, real_input_flag = schedule_sampling(eta, itr, args)

        #########################################################
        ######### train part code to be impoletented ############

        gen_imgs, ave_loss = model.train(ims, real_input_flag, exe, main_program)

        if itr%args.interval_print == 0:
            print(itr, "loss:{}".format(ave_loss))

        clone_program = main_program.clone(for_test=True)
        if itr%args.interval_test == 0:
            train_test(model,test_input_handle, clone_program, exe, args)

        if itr%args.interval_save == 0:
            model.save(itr, exe)
            # train_test(model,test_input_handle, clone_program, exe, args)

        # if itr%2000 == 0:
        #     test(args, model)
        # if itr%20000 == 0:
        #     model.save()

        ##########################################################
        ##########################################################

        train_input_handle.next()


def train_test(model, test_input_handle, clone_program,exe, args):
    """Evaluates a model."""
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'test...')
    test_input_handle.begin(do_shuffle=True)
    res_path = os.path.join(args.gen_frm_dir, str(args.save_name))
    if os.path.exists(res_path):
        shutil.rmtree(res_path)
    os.mkdir(res_path)
    avg_mse = 0
    batch_id = 0
    img_mse, ssim, psnr = [], [], []
    output_length = args.total_length - args.input_length

    for i in range(output_length):
        img_mse.append(0)
        ssim.append(0)
        psnr.append(0)

    real_input_flag_zero = np.zeros((args.batch_size, output_length - 1,
                                     args.img_height // args.patch_size,
                                     args.img_width // args.patch_size,
                                     args.patch_size ** 2 * args.img_channel), dtype='float32')

    while not test_input_handle.no_batch_left():
        batch_id = batch_id + 1
        test_ims = test_input_handle.get_batch()
        test_dat = preprocess.reshape_patch(test_ims, args.patch_size)
        # test_dat = np.split(test_dat, args.n_gpu)
        img_gen = model.test(test_dat, real_input_flag_zero, clone_program, exe)

        # Concat outputs of different gpus along batch
        # img_gen = np.concatenate(img_gen)
        img_gen = preprocess.reshape_patch_back(img_gen, args.patch_size)
        img_out = img_gen[:, -output_length:]
        target_out = test_ims[:, -output_length:]
        # MSE per frame
        for i in range(output_length):
            x = target_out[:, i]
            gx = img_out[:, i]
            gx = np.maximum(gx, 0)
            gx = np.minimum(gx, 1)
            mse = np.square(x - gx).sum()
            img_mse[i] += mse
            avg_mse += mse
            # for b in range(configs.batch_size):
            #     ssim[i] += compare_ssim(x[b], gx[b], multichannel=True)
            x = np.uint8(x * 255)
            gx = np.uint8(gx * 255)
            psnr[i] += batch_psnr(gx, x)

        # save prediction examples
        if batch_id <= args.num_save_samples:
            path = os.path.join(res_path, str(batch_id))
            os.mkdir(path)
            for i in range(args.total_length):
                if (i + 1) < 10:
                    name = 'gt0' + str(i + 1) + '.png'
                else:
                    name = 'gt' + str(i + 1) + '.png'
                file_name = os.path.join(path, name)
                img_gt = np.uint8(test_ims[0, i] * 255)
                cv2.imwrite(file_name, img_gt)
            for i in range(output_length):
                if (i + args.input_length + 1) < 10:
                    name = 'pd0' + str(i + args.input_length + 1) + '.png'
                else:
                    name = 'pd' + str(i + args.input_length + 1) + '.png'
                file_name = os.path.join(path, name)
                img_pd = img_gen[0, i]
                img_pd = np.maximum(img_pd, 0)
                img_pd = np.minimum(img_pd, 1)
                img_pd = np.uint8(img_pd * 255)
                cv2.imwrite(file_name, img_pd)
        test_input_handle.next()
        if batch_id > args.max_iterations_test:
            break

    avg_mse = avg_mse / (batch_id * args.batch_size * args.n_gpu)
    print('mse per seq: ' + str(avg_mse))
    for i in range(output_length):
        print(img_mse[i] / (batch_id * args.batch_size * args.n_gpu))

    psnr = np.asarray(psnr, dtype=np.float32) / batch_id
    print('psnr per frame: ' + str(np.mean(psnr)))
    for i in range(output_length):
        print(psnr[i])


def infer(args):
    if args.train_data_paths is None or args.valid_data_paths is None:
        raise ValueError("the data paths mush be given !!")
    # load data
    test_input_handle = datasets_factory.data_provider(
        args.dataset_name,
        args.train_data_paths,
        args.valid_data_paths,
        args.batch_size * args.n_gpu,
        args.img_width,
        seq_length=args.total_length,
        is_training=False)

    #########################################################
    ######### test part code to be impoletented ############
    #res_train = model.test(ims, real_input_flag, exe, place)
    ##########################################################
    ##########################################################

    """Evaluates a model by inferring."""
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'test...')
    test_input_handle.begin(do_shuffle=False)
    if not os.path.exists(args.infer_result_dir):
        os.mkdir(args.infer_result_dir)
    avg_mse = 0
    batch_id = 0
    img_mse, ssim, psnr = [], [], []
    output_length = args.total_length - args.input_length

    for i in range(output_length):
        img_mse.append(0)
        ssim.append(0)
        psnr.append(0)

    real_input_flag_zero = np.zeros((args.batch_size, output_length - 1,
                                     args.img_width // args.patch_size,
                                     args.img_width // args.patch_size,
                                     args.patch_size ** 2 * args.img_channel), dtype='float32')

    place = fluid.CUDAPlace(0) if args.use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)

    # inference_scope = fluid.core.Scope()
    # with fluid.scope_guard(inference_scope):
        # Use fluid.io.load_inference_model to obtain the inference program desc,
        # the feed_target_names (the names of variables that will be feeded
        # data using feed operators), and the fetch_targets (variables that
        # we want to obtain data from using fetch operators).
    [inference_program, feed_target_names,
     fetch_targets] = fluid.io.load_inference_model(
        args.load_dir, exe, None, None)

        # Construct feed as a dictionary of {feed_target_name: feed_target_data}
        # and results will contain a list of data corresponding to fetch_targets.
        # results = exe.run(
        #     inference_program,
        #     feed={feed_target_names[0]: tensor_img},
        #     fetch_list=fetch_targets)
        # lab = numpy.argsort(results)
        # print("Inference result of image/infer_3.png is: %d" % lab[0][0][-1])

    while not test_input_handle.no_batch_left():
        batch_id = batch_id + 1
        test_ims = test_input_handle.get_batch()
        test_dat = preprocess.reshape_patch(test_ims, args.patch_size)
        #test_dat = np.split(test_dat, args.n_gpu)
        # img_gen = .infer(test_dat, real_input_flag_zero, exe, place)
        imgs = exe.run(inference_program, feed={feed_target_names[0]: test_dat,
                                                feed_target_names[1]: real_input_flag_zero},
                       fetch_list=fetch_targets)
        img_gen = imgs[0]
        # Concat outputs of different gpus along batch
        # img_gen = np.concatenate(img_gen)
        img_gen = preprocess.reshape_patch_back(img_gen, args.patch_size)
        img_out = img_gen[:, -output_length:]
        target_out = test_ims[:, -output_length:]
        # MSE per frame
        for i in range(output_length):
            x = target_out[:, i]
            gx = img_out[:, i]
            gx = np.maximum(gx, 0)
            gx = np.minimum(gx, 1)
            mse = np.square(x - gx).sum()
            img_mse[i] += mse
            avg_mse += mse
            # for b in range(args.batch_size):
            #     ssim[i] += compare_ssim(x[b], gx[b], multichannel=True)
            x = np.uint8(x * 255)
            gx = np.uint8(gx * 255)
            psnr[i] += batch_psnr(gx, x)

        # save prediction examples
        if batch_id <= args.num_save_samples:
            path = os.path.join(args.infer_result_dir, str(batch_id))
            os.mkdir(path)
            for i in range(args.total_length):
                if (i + 1) < 10:
                    name = 'gt0' + str(i + 1) + '.png'
                else:
                    name = 'gt' + str(i + 1) + '.png'
                file_name = os.path.join(path, name)
                img_gt = np.uint8(test_ims[0, i] * 255)
                cv2.imwrite(file_name, img_gt)
            for i in range(output_length):
                if (i + args.input_length + 1) < 10:
                    name = 'pd0' + str(i + args.input_length + 1) + '.png'
                else:
                    name = 'pd' + str(i + args.input_length + 1) + '.png'
                file_name = os.path.join(path, name)
                img_pd = img_gen[0, i]
                img_pd = np.maximum(img_pd, 0)
                img_pd = np.minimum(img_pd, 1)
                img_pd = np.uint8(img_pd * 255)
                cv2.imwrite(file_name, img_pd)
        test_input_handle.next()
        if batch_id > args.max_iterations_test:
            break

    avg_mse = avg_mse / (batch_id * args.batch_size * args.n_gpu)
    print('mse per seq: ' + str(avg_mse))
    for i in range(output_length):
        print(img_mse[i] / (batch_id * args.batch_size * args.n_gpu))

    psnr = np.asarray(psnr, dtype=np.float32) / batch_id
    print('psnr per frame: ' + str(np.mean(psnr)))
    for i in range(output_length):
        print(psnr[i])


def print_model_info():
    pass


def schedule_sampling(eta, itr, args):
    """Gets schedule sampling parameters for training."""
    zeros = np.zeros(
        (args.batch_size, args.total_length - args.input_length - 1,
         args.img_width // args.patch_size, args.img_width // args.patch_size,
         args.patch_size ** 2 * args.img_channel), dtype='float32')
    if not args.scheduled_sampling:
        return 0.0, zeros

    if itr < args.sampling_stop_iter:
        eta -= args.sampling_changing_rate
    else:
        eta = 0.0
    random_flip = np.random.random_sample(
        (args.batch_size, args.total_length - args.input_length - 1))
    true_token = (random_flip < eta)
    ones = np.ones(
        (args.img_width // args.patch_size, args.img_width // args.patch_size,
         args.patch_size ** 2 * args.img_channel), dtype='float32')
    zeros = np.zeros(
        (args.img_width // args.patch_size, args.img_width // args.patch_size,
         args.patch_size ** 2 * args.img_channel), dtype='float32')
    real_input_flag = []
    for i in range(args.batch_size):
        for j in range(args.total_length - args.input_length - 1):
            if true_token[i, j]:
                real_input_flag.append(ones)
            else:
                real_input_flag.append(zeros)
    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(
        real_input_flag,
        (args.batch_size, args.total_length - args.input_length - 1,
         args.img_width // args.patch_size, args.img_width // args.patch_size,
         args.patch_size ** 2 * args.img_channel))
    return eta, real_input_flag


def main():
    parser = argparse.ArgumentParser("please give appropriate arguments")
    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--model_name', default='e3d_lstm', type=str)
    parser.add_argument('--lstm', default='ei', type=str)
    parser.add_argument('--dataset_name', default='action', type=str)  #action
    parser.add_argument('--gen_frm_dir', default='./gen_frm', type=str)
    parser.add_argument('--save_dir', default='./checkpoints', type=str)
    parser.add_argument('--load_dir', default='./checkpoints/inference.model-1800')
    parser.add_argument('--save_name', default='save', type=str)
    parser.add_argument('--infer_result_dir', default='infer_result', type=str)
    parser.add_argument('--train_data_paths', default=None, type=str)
    parser.add_argument('--valid_data_paths', default=None, type=str)
    parser.add_argument('--loss_l1', default='1', type=str)
    parser.add_argument('--n_gpu', default=1, type=int)
    parser.add_argument('--interval_print', default=10, type=int)
    parser.add_argument('--interval_test', default=1000, type=int)
    parser.add_argument('--interval_save', default=10000, type=int)
    parser.add_argument('--use_cuda', default=1, type=int)
    parser.add_argument('--epoch', default=10, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--num_hidden', default='16,16', type=str)
    parser.add_argument('--seq_length', default=5, type=int)
    parser.add_argument('--input_length', default=10, type=int)
    parser.add_argument('--total_length', default=20, type=int)
    parser.add_argument('--img_width', default=64, type=int)
    parser.add_argument('--img_height', default=64, type=int)
    parser.add_argument('--patch_size', default=8, type=int)
    parser.add_argument('--img_channel', default=1, type=int)
    parser.add_argument('--scheduled_sampling', default=True, type=bool)
    parser.add_argument('--sampling_stop_iter', default=100000, type=int)
    parser.add_argument('--sampling_changing_rate', default=0.00001, type=float)
    parser.add_argument('--sampling_start_value', default=1.0, type=float)
    parser.add_argument('--num_save_samples', default=200, type=int)
    parser.add_argument('--max_iterations_test', default=500, type=int)
    parser.add_argument('--max_iterations', default=200000, type=int)

    args = parser.parse_args()

    # setattr(args, 'train_data_paths', "../kth_action")
    # setattr(args, 'valid_data_paths', "../bai/kth_action")
    if args.dataset_name == 'mnist':
        setattr(args, 'train_data_paths', "../moving_mnist_example/moving-mnist-train.npz")
        setattr(args, 'valid_data_paths', "../moving_mnist_example/moving-mnist-valid.npz")
        args.img_width = 64
        args.img_height = 64
    else:
        setattr(args, 'train_data_paths', "../kth_action")
        setattr(args, 'valid_data_paths', "../kth_action")
        args.img_width = 128
        args.img_height = 128

    print(args)

    if args.mode is not None:
        if args.mode == 'train':
            model = Model(args)
            train(args, model)
        elif args.mode == 'infer':
            infer(args)
        elif args.mode == 'print':
            print_model_info()
        else:
            raise ValueError("the mode is not right!!")


if __name__ == '__main__':
    main()
