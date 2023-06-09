import torch
from p2mUtils.models import Trainer
from p2mUtils.utils import process_output
from p2mUtils.viz import plot_to_tensorboard , image_to_tensorboard
from pytorch_lightning.utilities import argparse
from s2mModel import GCN
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
import simpleRotoDataset
from simpleRotoDataset import ellipsoid
from datetime import datetime
from torch.utils.data import random_split
import os
import numpy as np
import pickle
from torch.utils.tensorboard import SummaryWriter
import multiprocessing
import glob
from torchvision.transforms import transforms, ToPILImage



def find_latest_checkpoint(checkpoint_dir):
    all_checkpoints = glob.glob(f'{checkpoint_dir}/**/*.pt', recursive=True)
    if not all_checkpoints:
        return None
    latest_checkpoint = max(all_checkpoints, key=os.path.getctime)
    return latest_checkpoint



args = argparse.ArgumentParser()
args.add_argument('--training_data',
                  help='Training data.',
                  type=str,
                  default='data/training_data/trainer_res.txt')
args.add_argument('--testing_data',
                  help='Testing data.',
                  type=str,
                  default='data/testing_data/test_list.txt')
args.add_argument('--batch_size', help='Batch size.', type=int, default=64) #was64
args.add_argument('--learning_rate',
                  help='Learning rate.',
                  type=float,
                  default=4e-5)
args.add_argument('--learning_rate_decay',
                  help='Learning rate.',
                  type=float,
                  default=0.97)
args.add_argument('--learning_rate_every',
                  help='Learning rate.',
                  type=int,
                  default=1)
args.add_argument('--show_every',
                  help='Frequency of displaying loss',
                  type=int,
                  default=2)
args.add_argument('--weight_decay',
                  help='Weight decay for L2 loss.',
                  type=float,
                  default=1e-5)
args.add_argument('--epochs',
                  help='Number of epochs to train.',
                  type=int,
                  default=200)
args.add_argument('--cnn_type',
                  help='Type of Neural Network',
                  type=str,
                  default='RES')
# args.add_argument('--checkpoint',
#                   help='Checkpoint to use.',
#                   type=str,
#                   default=r'D:\pyG\temp\RES\05-02_08-12-45\epoch_1\last_checkpoint.pt'
#                  )  # rechanged #changed
args.add_argument('--info_ellipsoid',
                  help='Initial Ellipsoid info',
                  type=str,
                  default='data/ellipsoid/info_ellipsoid.dat')
args.add_argument('--hidden',
                  help='Number of units in  hidden layer.',
                  type=int,
                  default=2048)
args.add_argument('--feat_dim',
                  help='Number of units in perceptual feature layer.',
                  type=int,
                  default=962)
args.add_argument('--coord_dim',
                  help='Number of units in output layer.',
                  type=int,
                  default=2)
args.add_argument('--tan_weight',
                  help='weight of tangent loss.',
                  type=float,
                  default=0.1)
args.add_argument('--point_weight', help='weight of point loss.',
                  type=float,
                  default=1)
args.add_argument('--chamfer_weight', help='weight of chamfer loss.', type=float, default=0.0)
args.add_argument('--align_weight', help='weight of align loss.', type=float, default=0.1)
args.add_argument('--surface_weight', help='weight of surface loss.', type=float, default=0.0)
args.add_argument('--df_weight', help='weight of distance field loss.', type=float, default=0.5)
args.add_argument('--dice_weight', help='weight of dice loss.', type=float, default=0.0)
args.add_argument('--sample_percentage',
                  help='Percentage of the dataset to use for training and testing.',
                  type=float,
                  default=0.5)
args.add_argument('--train_percentage',
                  help='Percentage of the dataset to use for training.',
                  type=float,
                  default=0.99)
args.add_argument('--num_workers',
                  help='Number of workers for data loading.',
                  type=int,
                  default=4)

args.add_argument('--checkpoint',
                  help='Checkpoint to use.',
                  type=str,
                  default=find_latest_checkpoint(r'D:\pyG\temp\RES'))
                  #default=False)



FLAGS = args.parse_args()

def main(FLAGS):
    #global writer, args, f, dataset, indices, ellipse, ellipse
    writer = SummaryWriter()
    use_cuda = torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True

    # get data loader
    # get input ellipse splines
    # create model wih input splines
    # train model
    mydir = os.path.join(os.getcwd(), 'temp', FLAGS.cnn_type,
                         datetime.now().strftime('%m-%d_%H-%M-%S'))
    os.makedirs(mydir)
    # dump the contents of args into a text file so we have a record of hyperparameters
    with open(os.path.join(mydir, 'args.txt'), 'w') as f:
        f.write(str(FLAGS))
    dataset = simpleRotoDataset.SimpleRotoDataset(root=r'D:\pyG\data\points\120423_183451_rev',
                                                  labelsJson="points120423_183451_rev.json")
    sample_percentage = FLAGS.sample_percentage
    if sample_percentage > 1.0 or sample_percentage <= 0.0:
        raise ValueError("Sample percentage must be between 0 and 1.")
    total_data = len(dataset)
    sample_data_size = int(total_data * sample_percentage)
    indices = np.random.choice(total_data, sample_data_size, replace=False)
    sampled_dataset = torch.utils.data.Subset(dataset, indices)
    train_percentage = FLAGS.train_percentage
    if train_percentage > 1.0 or train_percentage <= 0.0:
        raise ValueError("Training percentage must be between 0 and 1.")
    train_data_size = int(sample_data_size * train_percentage)
    test_data_size = sample_data_size - train_data_size
    train_data, test_data = random_split(sampled_dataset, [train_data_size, test_data_size])
    data_loader = DataLoader(train_data, batch_size=FLAGS.batch_size, shuffle=True, pin_memory=False,
                             num_workers=FLAGS.num_workers)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, pin_memory=False)
    ellipse = simpleRotoDataset.ellipsoid2(npoints=5)
    # normalise ellipse
    ellipse.shape.data.x = ellipse.shape.data.x / 224
    if use_cuda:
        ellipse.shape.x = ellipse.shape.x.cuda()
        ellipse.shape.data = ellipse.shape.data.cuda()
    model = GCN(ellipse, FLAGS, writer)
    if FLAGS.checkpoint:
        if use_cuda:
            model.load_state_dict(torch.load(FLAGS.checkpoint), strict=False)
            model = model.cuda()
            print(f"Model loaded on GPU {torch.cuda.get_device_name(0)} from {FLAGS.checkpoint}")
        else:
            model.load_state_dict(torch.load(FLAGS.checkpoint, map_location='cpu'), strict=False)
            print(f"Model loaded on CPU from {FLAGS.checkpoint}")
    else:
        print("No checkpoint provided or found, training from scratch.")

    trainer = Trainer(ellipse, model, FLAGS, writer)
    if use_cuda:
        model = model.cuda()
    train_number = len(train_data)
    print('---- Loadind training data, {} num samples'.format(train_number))
    print('\n')
    starter = datetime.now()
    for epoch in range(FLAGS.epochs):
        data_iter = iter(data_loader)
        if (epoch + 1) % FLAGS.learning_rate_every == 0:
            trainer.decay_lr()
        start_epoch = datetime.now()
        timer = start_epoch
        epoch_dir = mydir + '/epoch_{}'.format(epoch + 1)
        os.makedirs(epoch_dir)
        os.makedirs(epoch_dir + '/outputs')
        print('-------- Folder created : {}'.format(epoch_dir))
        iterNum = int(train_number / FLAGS.batch_size)
        all_loss = np.zeros((iterNum, 3), dtype='float32')
        all_loss_test = np.zeros((train_number, 3), dtype='float32')
        print('-------- Training epoch {} ...'.format(epoch + 1))
        for iters in range(iterNum):
            start_iter = datetime.now()
            torch.cuda.empty_cache()
            image, spline, path, df = next(data_iter)
            image = image.float()
            if use_cuda:
                image = image.cuda()
                spline = spline.cuda()
                df = df.cuda()
            # dists,out1,out2,out3=trainer.optimizer_step(image,spline)

            # write input image and plot spline to tensorboard
            # create PIL image from tensor and show

            global_iters = (epoch * iterNum) + iters
            trainer.global_step = global_iters
            image_to_tensorboard(writer, global_iters, image[0], "input_image")
            plot_to_tensorboard(writer, global_iters, spline.cpu().numpy()[0][:, :2], "input_spline")

            # normalize spline
            spline = spline / 224

            dists, out1 = trainer.optimizer_step(image, spline, df)
            all_loss[iters] = dists
            mean_loss = np.mean(all_loss[:, 0][np.where(all_loss[:, 0])])
            mean_loss_point = np.mean(all_loss[:, 1][np.where(all_loss[:, 1])])
            mean_loss_tangent = np.mean(all_loss[:, 2][np.where(all_loss[:, 2])])

            # write losses to tensorboard
            writer.add_scalar('Loss/train', dists[0], global_iters)
            writer.add_scalar('Loss/train_point', dists[1], global_iters)
            writer.add_scalar('Loss/train_tangent', dists[2], global_iters)
            # plot output to tensorboard
            plot_to_tensorboard(writer, global_iters, out1)
            # write network graph
            # if iters==0 and epoch==0:
            #     writer.add_graph(model, input_to_model=(ellipse) , verbose=False)

            end_iter = datetime.now()
            if iters == 0:
                total_iter = end_iter - start_iter
                print(" REAL TIME PER IMAGE == ",
                      total_iter.seconds / FLAGS.batch_size)
            if (iters + 1) % FLAGS.show_every == 0:
                print(
                    '------------ Iteration = {}, mean loss = {:.2f}, iter loss = {:.2f}'
                    .format(iters + 1, mean_loss, dists[0]))

                print("Time for iterations :", datetime.now() - timer)
                timer = datetime.now()
                print("Global time :", timer - starter)
                print(f"global step {trainer.global_step}")
        # write loss at epoch end
        writer.add_scalar('Loss/train_epoch', mean_loss, epoch)
        print('-------- Training epoch {} done !'.format(epoch + 1))
        print("Time for epoch :", timer - start_epoch)
        print("Global time :", timer - starter)

        ckp_dir = epoch_dir + '/last_checkpoint.pt'
        torch.save(model.state_dict(), ckp_dir)
        print('-------- Training checkpoint last saved !')

        print('-------- Testing epoch {} ...'.format(epoch + 1))
        test_iter = iter(test_loader)
        for i in range(len(test_data)):
            image, spline, path, df = next(test_iter)
            df = df[0]
            # print(f"Testing image {i} from path {path} of {len(test_data)}")
            image = image.float()
            if use_cuda:
                image = image.cuda()
                spline = spline.cuda()
                df = df.cuda()
            output3 = model(image)
            shape = process_output(output3)
            # calculate loss
            # if batch feed in 1 spline/output at a time
            batch = False
            batchSize = 1
            if len(output3.shape) == 3:
                batch = True
                batchSize = output3.shape[0]
                output3 = output3.view(output3.shape[0] * output3.shape[1], -1)
                spline = spline.view(spline.shape[0] * spline.shape[1], -1)
            output3 = [output3[None]]
            # reverse normalize spline
            output3 *= 224
            dists = trainer._get_loss_pt(spline, output3, spline, 6, df)

            all_loss_test[i] = [dists[0].item(), dists[1].item(), dists[2].item()]
            # divide by batch size
            if batch:
                all_loss_test[i] = all_loss_test[i] / batchSize

            mean_loss_test = np.mean(all_loss_test[:, 0][np.where(all_loss_test[:, 0])])
            mean_loss_point_test = np.mean(all_loss_test[:, 1][np.where(all_loss_test[:, 1])])
            mean_loss_tangent_test = np.mean(all_loss_test[:, 2][np.where(all_loss_test[:, 2])])

            pred_path = epoch_dir + '/outputs/'
            # save list of point2D objects to file pickle format
            try:
                with open(pred_path + f"/{epoch}_{i}.pkl", "wb") as f:
                    pickle.dump([path, shape], f, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as ex:
                print("Error during pickling object (Possibly unsupported):", ex)

        print('-------- Testing epoch {} done !'.format(epoch + 1))
        # write the loss,point loss and tangent loss to tensorboard
        writer.add_scalar('Loss/test', mean_loss_test, epoch)
        writer.add_scalar('Loss/test_point', mean_loss_point_test, epoch)
        writer.add_scalar('Loss/test_tangent', mean_loss_tangent_test, epoch)
        print("Mean loss test:", mean_loss_test)
        print("Mean loss point test:", mean_loss_point_test)
        print("Mean loss tangent test:", mean_loss_tangent_test)

        print("")
        print('\n')




if __name__ == '__main__':
    multiprocessing.freeze_support()
    FLAGS = args.parse_args()
    main(FLAGS)