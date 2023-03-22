import torch
from p2mUtils.models import Trainer
from p2mUtils.utils import process_output
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

use_cuda = torch.cuda.is_available()
args = argparse.ArgumentParser()
args.add_argument('--training_data',
                  help='Training data.',
                  type=str,
                  default='data/training_data/trainer_res.txt')
args.add_argument('--testing_data',
                  help='Testing data.',
                  type=str,
                  default='data/testing_data/test_list.txt')
args.add_argument('--batch_size', help='Batch size.', type=int, default=1)
args.add_argument('--learning_rate',
                  help='Learning rate.',
                  type=float,
                  default=5e-5)
args.add_argument('--learning_rate_decay',
                  help='Learning rate.',
                  type=float,
                  default=0.97)
args.add_argument('--learning_rate_every',
                  help='Learning rate.',
                  type=int,
                  default=2)
args.add_argument('--show_every',
                  help='Frequency of displaying loss',
                  type=int,
                  default=10)
args.add_argument('--weight_decay',
                  help='Weight decay for L2 loss.',
                  type=float,
                  default=1e-5)
args.add_argument('--epochs',
                  help='Number of epochs to train.',
                  type=int,
                  default=20)
args.add_argument('--cnn_type',
                  help='Type of Neural Network',
                  type=str,
                  default='RES')
args.add_argument('--checkpoint',
                  help='Checkpoint to use.',
                  type=str,
                  default='temp/RES/03-21_18-28-31/epoch_20/last_checkpoint.pt'
                  )  # rechanged #changed
args.add_argument('--info_ellipsoid',
                  help='Initial Ellipsoid info',
                  type=str,
                  default='data/ellipsoid/info_ellipsoid.dat')
args.add_argument('--hidden',
                  help='Number of units in  hidden layer.',
                  type=int,
                  default=256)
args.add_argument('--feat_dim',
                  help='Number of units in perceptual feature layer.',
                  type=int,
                  default=962)
args.add_argument('--coord_dim',
                  help='Number of units in output layer.',
                  type=int,
                  default=2)

FLAGS = args.parse_args()

#get data loader

# get input ellipse splines

#create model wih input splines

#train model
mydir = os.path.join(os.getcwd(), 'temp', FLAGS.cnn_type,
                     datetime.now().strftime('%m-%d_%H-%M-%S'))
os.makedirs(mydir)
dataset=simpleRotoDataset.SimpleRotoDataset(root='D:/pyG/data/points/')
train_data, test_data = random_split(dataset, [0.9, 0.1])
data_loader=DataLoader(train_data,batch_size=1,shuffle=False)
test_loader=DataLoader(test_data,batch_size=1,shuffle=False)


ellipse=ellipsoid()
if use_cuda:
    ellipse.shape.x = ellipse.shape.x.cuda()
    ellipse.shape.data=ellipse.shape.data.cuda()
model=GCN(ellipse,FLAGS)
# if use_cuda:
#     model.load_state_dict(torch.load(FLAGS.checkpoint), strict=False)
#     model = model.cuda()
#     print(f"Model loaded on GPU {torch.cuda.get_device_name(0)} from {FLAGS.checkpoint}")
# else:
#     model.load_state_dict(torch.load(FLAGS.checkpoint, map_location='cpu'), strict=False)
#     print(f"Model loaded on CPU from {FLAGS.checkpoint}")


trainer = Trainer(ellipse, model, FLAGS)
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
    all_loss = np.zeros(int(train_number / FLAGS.batch_size), dtype='float32')
    print('-------- Training epoch {} ...'.format(epoch + 1))
    for iters in range(int(train_number / FLAGS.batch_size)):
        start_iter = datetime.now()
        torch.cuda.empty_cache()
        image,spline, path=next(data_iter)
        image=image.float()
        if use_cuda:
            image=image.cuda()
            spline=spline.cuda()
        #dists,out1,out2,out3=trainer.optimizer_step(image,spline)
        dists,out1=trainer.optimizer_step(image,spline)
        all_loss[iters] = dists
        mean_loss = np.mean(all_loss[np.where(all_loss)])
        end_iter = datetime.now()
        if iters == 0:
            total_iter = end_iter - start_iter
            print(" REAL TIME PER IMAGE == ",
                  total_iter.seconds / FLAGS.batch_size)
        if (iters + 1) % FLAGS.show_every == 0:
            print(
                '------------ Iteration = {}, mean loss = {:.2f}, iter loss = {:.2f}'
                .format(iters + 1, mean_loss, dists))

            print("Time for iterations :", datetime.now() - timer)
            timer = datetime.now()
            print("Global time :", timer - starter)

    print('-------- Training epoch {} done !'.format(epoch + 1))
    print("Time for epoch :", timer - start_epoch)
    print("Global time :", timer - starter)

    ckp_dir = epoch_dir + '/last_checkpoint.pt'
    torch.save(model.state_dict(), ckp_dir)
    print('-------- Training checkpoint last saved !')

    print('-------- Testing epoch {} ...'.format(epoch + 1))
    test_iter=iter(test_loader)
    for i in range(len(test_data)):
        image, spline ,path =next(test_iter)
        #print(f"Testing image {i} from path {path} of {len(test_data)}")
        image = image.float()
        if use_cuda:
            image = image.cuda()
            spline = spline.cuda()
        output3 = model(image)
        shape = process_output(output3)
        pred_path = epoch_dir + '/outputs/'
        #save list of point2D objects to file pickle format
        try:
            with open(pred_path+f"/{epoch}_{i}.pkl", "wb") as f:
                pickle.dump([path,shape], f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as ex:
            print("Error during pickling object (Possibly unsupported):", ex)

    print('-------- Testing epoch {} done !'.format(epoch + 1))
    print('\n')

