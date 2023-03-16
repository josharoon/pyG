import torch
from p2mUtils.models import Trainer
from pytorch_lightning.utilities import argparse
from s2mModel import GCN
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
import simpleRotoDataset
from simpleRotoDataset import ellipsoid



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
args.add_argument('--batch_size', help='Batch size.', type=int, default=10)
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
                  default='data/checkpoints/last_checkpoint_res.pt'
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
                  default=963)
args.add_argument('--coord_dim',
                  help='Number of units in output layer.',
                  type=int,
                  default=3)

FLAGS = args.parse_args()

#get data loader

# get input ellipse splines

#create model wih input splines

#train model
dataset=simpleRotoDataset.SimpleRotoDataset(root='D:/pyG/data/points/')
data_loader=DataLoader(dataset,batch_size=1,shuffle=False)
ellipse=ellipsoid()
if use_cuda:
    ellipse.shape.x.cuda()
model=GCN(ellipse,FLAGS)
trainer = Trainer(ellipse, model, FLAGS)
if use_cuda:
    model = model.cuda()
for iters in range(len(dataset)):
    torch.cuda.empty_cache()
    spline,image=next(iter(data_loader))
    image=image.float()
    if use_cuda:
        image=image.cuda()
        spline=spline.cuda()
        dists,out1,out2,out3=trainer.optimizer_step(image,spline)

