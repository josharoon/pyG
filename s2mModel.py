from __future__ import division
import torch
import torch.nn as nn
from p2mUtils.inits import *
from p2mUtils.layers import *
from p2mUtils.utils import *
from p2mUtils.viz import *


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.layers = []

    def _prepare(self, img_feat):
        for layer in self.proj_layers:
            layer._prepare(img_feat)

    def _build(self):
        raise NotImplementedError

    def build(self):
        self._build()
        # self.unpool_layers = []
        # self.unpool_layers.append(
        #     GraphPooling(tensor_dict=self.tensor_dict, pool_id=1))
        # self.unpool_layers.append(
        #     GraphPooling(tensor_dict=self.tensor_dict, pool_id=2))

    def forward(self, img_inp):
        #write image 2 tensorboard
        image_to_tensorboard(self.writer,0,img_inp[0],name="model_input")


        reshape = len(img_inp.shape) == 3
        if reshape:
            img_inp = img_inp.unsqueeze(0)
        inputs = get_features(self.ellipse, img_inp)
        #if inputs dim3 is 6 then reshape dim3 to 2 and dim2 should be 3 times bigger
        #if inputs.shape[2]==6:
        #    inputs=inputs.reshape(inputs.shape[0],inputs.shape[1]*3,2)
        #inputs should be 10x2 when we are just interested in xy coordinates

        #inputs=inputs[:,:,:2]
        #before project we need to add tangent handles to xy coordinates


        img_feat = self.forward_cnn(img_inp)
        featMaps=[]
        for f in img_feat:
            f=f[0]
            # sum over all channels
            f=f.sum(0)
            featMaps.append(f)
        for f in featMaps:
             image_to_tensorboard(self.writer,0,f,name=f"feature_map_{(f.shape[0])}x{(f.shape[1])}")

        self._prepare(img_feat)

        # Build sequential resnet model
        eltwise = [
            3, 5, 7, 9, 11, 13, 19, 21, 23, 25, 27, 29, 35, 37, 39, 41, 43, 45
        ]
        concat = [15, 31]

        activations = []
        activations.append(inputs)

        for idx, layer in enumerate(self.layers):
            hidden = layer(activations[-1])
            if idx in eltwise:
                hidden = torch.add(hidden, activations[-2]) * 0.5
            if idx in concat:
                hidden = torch.cat([hidden, activations[-2]], 2)
            activations.append(hidden)

        output1 = activations[15]
        # output1_2 = self.unpool_layers[0](output1)
        #
        # output2 = activations[31]
        # output2_2 = self.unpool_layers[1](output2)
        #
        # output3 = activations[-1]

        # if not reshape:
        #     return output1, output1_2, output2, output2_2, output3
        # else:
        #     return output1.squeeze(0), output1_2.squeeze(0), output2.squeeze(
        #         0), output2_2.squeeze(0), output3.squeeze(0)

        if not reshape:
            return output1
        else:
            return output1.squeeze(0)


class GCN(Model):

    def __init__(self, ellipse, args,writer=None):
        super(GCN, self).__init__()
        self.ellipse = ellipse
        self.args = args
        self.build()
        self.writer=writer

    def _build(self):
        FLAGS = self.args
        self.build_cnn()
        # first project block
        self.layers.append(GraphProjection())
        self.layers.append(
            GraphConvolution(input_dim=FLAGS.feat_dim,
                             output_dim=FLAGS.hidden,
                             gcn_block_id=1,
                             ellipsoid=self.ellipse))

        for _ in range(12):
            self.layers.append(
                GraphConvolution(input_dim=FLAGS.hidden,
                                 output_dim=FLAGS.hidden,
                                 gcn_block_id=1,
                                 ellipsoid=self.ellipse))
        self.layers.append(
            GraphConvolution(input_dim=FLAGS.hidden,
                             output_dim=FLAGS.coord_dim,
                             act=None,
                             gcn_block_id=1,
                             ellipsoid=self.ellipse))
#Start with 1st block

        # # second project block
        # self.layers.append(GraphProjection())
        # self.layers.append(GraphPooling(tensor_dict=self.ellipse,
        #                                 pool_id=1))  # unpooling
        # self.layers.append(
        #     GraphConvolution(input_dim=FLAGS.feat_dim + FLAGS.hidden,
        #                      output_dim=FLAGS.hidden,
        #                      gcn_block_id=2,
        #                      ellipsoid=self.ellipse))
        # for _ in range(12):
        #     self.layers.append(
        #         GraphConvolution(input_dim=FLAGS.hidden,
        #                          output_dim=FLAGS.hidden,
        #                          gcn_block_id=2,
        #                          ellipsoid=self.ellipse))
        # self.layers.append(
        #     GraphConvolution(input_dim=FLAGS.hidden,
        #                      output_dim=FLAGS.coord_dim,
        #                      act=None,
        #                      gcn_block_id=2,
        #                      ellipsoid=self.ellipse))
        # # third project block
        # self.layers.append(GraphProjection())
        # self.layers.append(GraphPooling(tensor_dict=self.ellipse,
        #                                 pool_id=2))  # unpooling
        # self.layers.append(
        #     GraphConvolution(input_dim=FLAGS.feat_dim + FLAGS.hidden,
        #                      output_dim=FLAGS.hidden,
        #                      gcn_block_id=3,
        #                      ellipsoid=self.ellipse))
        # for _ in range(12):
        #     self.layers.append(
        #         GraphConvolution(input_dim=FLAGS.hidden,
        #                          output_dim=FLAGS.hidden,
        #                          gcn_block_id=3,
        #                          ellipsoid=self.ellipse))
        # self.layers.append(
        #     GraphConvolution(input_dim=FLAGS.hidden,
        #                      output_dim=int(FLAGS.hidden / 2),
        #                      gcn_block_id=3,
        #                      ellipsoid=self.ellipse))
        # self.layers.append(
        #     GraphConvolution(input_dim=int(FLAGS.hidden / 2),
        #                      output_dim=FLAGS.coord_dim,
        #                      act=None,
        #                      gcn_block_id=3,
        #                      ellipsoid=self.ellipse))
        # self.layers = nn.ModuleList(self.layers)

        self.proj_layers = []
        for layer in self.layers:
            if layer.layer_type == 'GraphProjection':
                self.proj_layers.append(layer)



    def build_cnn(self):
        self.cnn_layers_01 = [
            nn.ZeroPad2d(1),
            nn.Conv2d(3, 16, 3, 1, padding=0),
            nn.ReLU()
        ]
        self.cnn_layers_02 = [
            nn.ZeroPad2d(1),
            nn.Conv2d(16, 16, 3, 1, padding=0),
            nn.ReLU()
        ]

        self.cnn_layers_11 = [
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.Conv2d(16, 32, 3, 2, padding=0),
            nn.ReLU()
        ]
        # 112 112
        self.cnn_layers_12 = [
            nn.ZeroPad2d(1),
            nn.Conv2d(32, 32, 3, 1, padding=0),
            nn.ReLU()
        ]
        self.cnn_layers_13 = [
            nn.ZeroPad2d(1),
            nn.Conv2d(32, 32, 3, 1, padding=0),
            nn.ReLU()
        ]

        self.cnn_layers_21 = [
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.Conv2d(32, 64, 3, 2, padding=0),
            nn.ReLU()
        ]
        # 56 56
        self.cnn_layers_22 = [
            nn.ZeroPad2d(1),
            nn.Conv2d(64, 64, 3, 1, padding=0),
            nn.ReLU()
        ]
        self.cnn_layers_23 = [
            nn.ZeroPad2d(1),
            nn.Conv2d(64, 64, 3, 1, padding=0),
            nn.ReLU()
        ]

        self.cnn_layers_31 = [
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.Conv2d(64, 128, 3, 2, padding=0),
            nn.ReLU()
        ]
        # 28 28
        self.cnn_layers_32 = [
            nn.ZeroPad2d(1),
            nn.Conv2d(128, 128, 3, 1, padding=0),
            nn.ReLU()
        ]
        self.cnn_layers_33 = [
            nn.ZeroPad2d(1),
            nn.Conv2d(128, 128, 3, 1, padding=0),
            nn.ReLU()
        ]

        self.cnn_layers_41 = [
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(128, 256, 5, 2, padding=0),
            nn.ReLU()
        ]
        # 14 14
        self.cnn_layers_42 = [
            nn.ZeroPad2d(1),
            nn.Conv2d(256, 256, 3, 1, padding=0),
            nn.ReLU()
        ]
        self.cnn_layers_43 = [
            nn.ZeroPad2d(1),
            nn.Conv2d(256, 256, 3, 1, padding=0),
            nn.ReLU()
        ]

        self.cnn_layers_51 = [
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(256, 512, 5, 2, padding=0),
            nn.ReLU()
        ]
        # 7 7
        self.cnn_layers_52 = [
            nn.ZeroPad2d(1),
            nn.Conv2d(512, 512, 3, 1, padding=0),
            nn.ReLU()
        ]
        self.cnn_layers_53 = [
            nn.ZeroPad2d(1),
            nn.Conv2d(512, 512, 3, 1, padding=0),
            nn.ReLU()
        ]
        self.cnn_layers_54 = [
            nn.ZeroPad2d(1),
            nn.Conv2d(512, 512, 3, 1, padding=0),
            nn.ReLU()
        ]

        keys = list(self.__dict__.keys())[:]
        for key in keys:
            if key.startswith('cnn_layers_'):
                setattr(self, key, nn.Sequential(*getattr(self, key)))

    def forward_cnn(self, img_inp):
        x = img_inp

        x = self.cnn_layers_01(x)
        x = self.cnn_layers_02(x)
        x0 = x

        x = self.cnn_layers_11(x)
        x_res = x
        x = self.cnn_layers_12(x)
        x = self.cnn_layers_13(x) + x_res
        x1 = x

        x = self.cnn_layers_21(x)
        x_res = x
        x = self.cnn_layers_22(x)
        x = self.cnn_layers_23(x) + x_res
        x2 = x

        x = self.cnn_layers_31(x)
        x_res = x
        x = self.cnn_layers_32(x)
        x = self.cnn_layers_33(x) + x_res
        x3 = x

        x = self.cnn_layers_41(x)
        x_res = x
        x = self.cnn_layers_42(x)
        x = self.cnn_layers_43(x) + x_res
        x4 = x

        x = self.cnn_layers_51(x)
        x_res = x
        x = self.cnn_layers_52(x)
        x = self.cnn_layers_53(x) + x_res
        x = self.cnn_layers_54(x)
        x5 = x

        img_feat = [x2, x3, x4, x5]
        #print(img_feat[0].shape, img_feat[1].shape, img_feat[2].shape, img_feat[3].shape)
        return img_feat

