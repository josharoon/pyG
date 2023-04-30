import random

import torch.optim as optim

from p2mUtils.miouSpline import splineArray2Image
from torch import nn
from torchvision.transforms import PILToTensor

from .losses import *
from .utils import *
from .chamfer import nn_distance_function as chamfer_dist
from dfUtils.cubicCurvesUtil import *
from p2mUtils.viz import *
from torchmetrics.classification import dice
use_cuda = torch.cuda.is_available()


class Trainer:

    def __init__(self, ellipse, network, args,writer=None):
        self.global_step = 0
        self.args = args
        self.network = network

        self.optimizer = optim.Adam(network.parameters(),
                                    lr=self.args.learning_rate,
                                    weight_decay=self.args.weight_decay)
        self.ellipse = ellipse
        self._get_loss = self._get_loss_pt
        self.ncoords=args.coord_dim
        self.tanWeight=args.tan_weight
        self.pointWeight=args.point_weight
        self.chamferWeight=args.chamfer_weight
        self.diceWeight=args.dice_weight
        self.dfWeight=args.df_weight
        self.alignWeight=args.align_weight
        self.surfWeight=args.surface_weight
        self.writer=writer

    # def get_loss(self, img_inp, labels):
    #     if type(img_inp) != list:
    #         inputs = get_features(self.tensor_dict, img_inp)
    #         outputs = self.network(img_inp)
    #     else:
    #         inputs = get_features(self.tensor_dict, img_inp[0])
    #         outputs = self.network(img_inp[0].unsqueeze(0),
    #                                img_inp[1].unsqueeze(0))
    #         outputs = [output.squeeze(0) for output in outputs]
    #     loss = self._get_loss(inputs, outputs, labels)
    #     return loss, outputs[0], outputs[2], outputs[4]

    def decay_lr(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= self.args.learning_rate_decay

    def get_loss(self, img_inp, labels,ncoords,df):
        if type(img_inp) != list:
            batch = len(img_inp.shape) == 4
            inputs = get_features(self.ellipse, img_inp) #return data from ellipse
            outputs = self.network(img_inp)

        else:
            batch = len(img_inp[0].shape) == 4
            inputs = get_features(self.ellipse, img_inp[0]) #return data from ellipse
            outputs = self.network(img_inp[0], img_inp[1])
        if batch:
            losses=[0,0,0,0]
            for idx, (input, label,dfi) in enumerate(zip(inputs, labels,df)):
                #output = [out[idx] for out in outputs]

                output=[outputs] #we only have one output at the moment
                itLosses=self._get_loss(input, output, label,ncoords,dfi)
                losses[0]+=itLosses[0]
                losses[1]+=itLosses[1]
                losses[2]+=itLosses[2]
                losses[3]+=itLosses[3]

            losses[0]/=len(inputs)
            losses[1]/=len(inputs)
            losses[2]/=len(inputs)
            losses[3]/=len(inputs)

            # loss /= len(inputs)
            # loss1 /= len(inputs)
            # loss2 /= len(inputs)
        else:
            losses = self._get_loss(inputs, outputs, labels, ncoords)
            # loss = self._get_loss(inputs, outputs, labels,ncoords)
            # loss1 = self._get_loss(inputs, outputs, labels,ncoords)[1]
            # loss2 = self._get_loss(inputs, outputs, labels,ncoords)[2]
        #return loss, outputs[0], outputs[2], outputs[4]
        return [losses[0],losses[1],losses[2],losses[3]], outputs[0]

    def _get_loss_tf(self, inputs, outputs, labels):
        #output1, output1_2, output2, output2_2, output3 = outputs
        loss = 0
        loss += mesh_loss(outputs[0], labels, self.ellipse, 1)
        loss += mesh_loss(outputs[2], labels, self.ellipse, 2)
        loss += mesh_loss(outputs[4], labels, self.ellipse, 3)
        loss += .1 * laplace_loss(inputs, outputs[0], self.ellipse, 1)
        loss += laplace_loss(outputs[1], outputs[2], self.ellipse, 2)
        loss += laplace_loss(outputs[3], outputs[4], self.ellipse, 3)
        for layer in self.network.layers:
            if layer.layer_type == 'GraphConvolution':
                for key, var in layer.vars.items():
                    loss += self.args.weight_decay * torch.sum(var**2)
        return loss

    def _get_loss_pt(self, inputs, outputs, labels, nCoords,df):
        # Edge Loss
        def edge_loss_pt(pred, labels, ellipse, block_id):
            gt_pts = labels[:, :2]
            idx1 = torch.t(ellipse.shape.edge_index)[:, 0]
            idx2 = torch.t(ellipse.shape.edge_index)[:, 0]

            if use_cuda:
                idx1 = idx1.cuda()
                idx2 = idx2.cuda()
            nod1 = torch.index_select(pred, 0, idx1)
            nod2 = torch.index_select(pred, 0, idx2)
            edge = nod1 - nod2
            # edge length loss
            edge_length = torch.sum(torch.pow(edge, 2), 1)
            edge_loss = torch.mean(edge_length) * 300
            return edge_loss

        def pointMatchLoss(shape1, shape2, nCoords):
            # shape1 and shape2 are tensors of shape (K,2)
            # shape1 = shape1.cpu().detach().numpy()
            # shape2 = shape2.cpu().detach().numpy()
            K = shape1.shape[0]
            assert K == shape2.shape[0]
            sumNorm = 0
            # tensor of zeros shape (K,2)
            Norms = torch.zeros(K, nCoords)
            for jInd in range(K):
                L1Norm = 0
                for iInd in range(K):
                    p2i = (iInd + jInd) % K
                    #print(f"j={jInd}, i={iInd}, p2Index={p2i}")
                    L1Norm += abs((shape1[iInd] - shape2[p2i]))
                    #print(f"L1Norm = {L1Norm}")
                Norms[jInd] = L1Norm
            #print(f"Norms = {Norms}")
            # sum the x and y components of the norms then grab the minimum
            sumNorm = torch.sum(Norms, axis=1)
            # print(f"sumNorm = {sumNorm}")
            minNorm = torch.min(sumNorm)
            #print(f"minNorm = {minNorm}")
            return minNorm

        def laplace_coord(input, ellipse, block_id):

            # Inputs :
            # input : nodes Tensor, size (n_pts, n_features)
            # lap_idx : laplace index matrix Tensor, size (n_pts, 10)
            #
            # Returns :
            # The laplacian coordinates of input with respect to edges as in lap_idx

            vertex = torch.cat(
                (input,
                 torch.zeros(1, 2).cuda()), 0) if use_cuda else torch.cat(
                     (input, torch.zeros(1, 2)), 0)

            #indices = ellipse['lape_idx'][block_id - 1][:, :8].long()
            indices = ellipse.lapIndex[:, :8].long()
            weights = ellipse.lapIndex[:, -1].float()
            #get laplacian maybe not working - only 2 coords not 5.
            if use_cuda:
                indices = indices.cuda()
                weights = weights.cuda()

            #weights = torch.reciprocal(weights).reshape((-1, 1)).repeat((1, 3))
            weights = torch.reciprocal(weights).reshape((-1, 1)).repeat((1, 2))
            num_pts, num_indices = indices.shape[0], indices.shape[1]
            indices = indices.reshape((-1,))
            vertices = torch.index_select(vertex, 0, indices)
            vertices = vertices.reshape((num_pts, num_indices, 2))

            laplace = torch.sum(vertices, 1)
            laplace = input - torch.mul(laplace, weights)

            return laplace

        def laplace_loss(input1, input2, ellipse, block_id):

            # Inputs :
            # input1, input2 : nodes Tensor before and after the deformation
            # lap_idx : laplace index matrix Tensor, size (n_pts, 10)
            # block_id : id of the deformation block (if different than 1 then adds
            # a move loss as in the original TF code)
            #
            # Returns :
            # The Laplacian loss, with the move loss as an additional term when relevant

            lap1 = laplace_coord(input1, ellipse, block_id)
            lap2 = laplace_coord(input2, ellipse, block_id)
            laplace_loss = torch.mean(torch.sum(torch.pow(lap1 - lap2, 2),
                                                1)) * 1500
            move_loss = torch.mean(torch.sum(torch.pow(input1 - input2, 2),
                                             1)) * 100

            if block_id == 1:
                return laplace_loss
            else:
                return laplace_loss + move_loss
            # Chamfer Loss

        def discrepancy_l2_squared(distance_field_A, distance_field_B):
            return (distance_field_A - distance_field_B) ** 2

        def distance_field_loss(distance_field_A, distance_field_B):
            vol_S = distance_field_A.numel()
            loss = th.sum(discrepancy_l2_squared(distance_field_A, distance_field_B)) / vol_S
            return loss
        def get_distancefield_loss(input1, input2, input2_df):
            #convert input 1 to correct format by subtracting main points from tangent handels

            # input1[:, 2:4] = input1[:, 2:4] - input1[:, :2]
            # input1[:, 4:6] = input1[:, 4:6] - input1[:, :2]
            # # input2[:, 2:4] = input2[:, 2:4] - input2[:, :2]
            # # input2[:, 4:6] = input2[:, 4:6] - input2[:, :2]

            #get control points
            Input_1_control_points=convert_to_cubic_control_points(input1[None,:]).to(torch.float64)

            source_points=create_grid_points(224,0,250,0,250).to(torch.float64)
            #get distance field
            distance_field = distance_to_curves(source_points, Input_1_control_points, 224).view(224, 224)
            distance_field = th.flip(distance_field, (1,))
            #normalize distance field to match input2_df
            dmax=torch.max(distance_field)
            distance_field=distance_field/dmax
            loss = distance_field_loss(distance_field, input2_df)

            # with a 1 in 100 random chance plot the distance field
            if random.randint(0, 100) == 1:
                Input_2_control_points = convert_to_cubic_control_points(input2[None, :]).to(torch.float64)
                # plot the distance field
                plot_distance_field(distance_field,1,"output")
                plot_distance_field(input2_df,1,"GT")

                plotCubicSpline(Input_1_control_points)
                plotCubicSpline(Input_2_control_points)

                input2_df=distance_to_curves(source_points, Input_2_control_points, 224).view(224, 224)
                input2_df = th.flip(input2_df, (1,))
                dmax = torch.max(input2_df)
                input2_df = input2_df / dmax
                plot_distance_field(input2_df,1,"GT_from_input2")


            return loss,distance_field

        def compute_alignment_fields(distance_fields):
            """Compute alignment unit vector fields from distance fields."""
            dx = distance_fields[..., 2:, 1:-1] - distance_fields[..., :-2, 1:-1]
            dy = distance_fields[..., 1:-1, 2:] - distance_fields[..., 1:-1, :-2]
            alignment_fields = th.stack([dx, dy], dim=-1)
            return alignment_fields / th.sqrt(th.sum(alignment_fields ** 2, dim=-1, keepdims=True) + 1e-6)

        def compute_occupancy_fields(distance_fields, eps=(2 / 128) ** 2):
            """Compute smooth occupancy fields from distance fields."""
            occupancy_fields = 1 - th.clamp(distance_fields / eps, 0, 1)
            return occupancy_fields ** 2 * (3 - 2 * occupancy_fields)

        pt_chamfer_loss = 0.
        pt_edge_loss = 0.
        pt_lap_loss = 0.
        blended_loss = 0.
        point_loss = 0.
        tangent_loss = 0.
        diceLossVal = 0.
        dfLossVal = 0.
        lap_const = [0.2, 1., 1.]
        idx=0 #we only have block 1 at the moment
        # for idx, (output, feat) in enumerate(
        #         zip([outputs[0], outputs[2], outputs[4]],
        #             [inputs, outputs[1], outputs[3]])):

        #for idx, (output, feat) in enumerate(outputs,inputs):

        output=outputs[0][0]
        # #resize output to 3x224x224
        # output=output.resize(3,224,224)
        #
        # feat=inputs[:,:2]
        # outImage=splineArray2Image(output)
        # #keep gradeint for dice loss
        # #convert out to float
        # outImage=outImage.float()
        #
        # labelImage=splineArray2Image(labels)
        # #convert label to float
        # labelImage=labelImage.float()
        # #now resize output to 3x224x224
        #
        #
        #
        #
        # #diceC = dice.Dice()
        # #calculate dice  loss between the 2 images
        # #diceLossVal += diceC(outImage,labelImage)
        # bce_loss = torch.nn.BCELoss()
        # diceLossVal += bce_loss(outImage,labelImage)

        #reshape output to 5x6 tensor

        output=output.resize(5,6)
        #calculate dice loss



        dist1, dist2, _, _ = chamfer_dist(output, labels[:, :6])

        #calculate distance field for output
        dfloss, dfOut = get_distancefield_loss(output, labels[:, :6], df)
        dfLossVal += dfloss
        alignment_fields=compute_alignment_fields(dfOut)
        occupancy_fields=compute_occupancy_fields(dfOut)
        target_occupancy_fields = compute_occupancy_fields(df)
        target_alignment_fields = compute_alignment_fields(df)
        surfaceloss = th.mean(target_occupancy_fields * dfOut + df * occupancy_fields)
        alignmentloss = th.mean(1 - th.sum(target_alignment_fields * alignment_fields, dim=-1) ** 2)


        print(f"dfLossVal: {dfLossVal}")


        pt_chamfer_loss += torch.mean(dist1) + torch.mean(dist2)
        #pt_edge_loss += edge_loss_pt(output, labels, self.ellipse,idx + 1)
        point_lossTemp =pointMatchLoss(output[:, :2], labels[:, :2],2)
        tangent_lossTemp = pointMatchLoss(output[:, 2:], labels[:, 2:],4)
        blended_loss += (point_lossTemp*self.pointWeight + tangent_lossTemp*self.tanWeight +dfLossVal*self.dfWeight+ surfaceloss*self.surfWeight + alignmentloss*self.alignWeight+self.diceWeight*diceLossVal+self.chamferWeight*pt_chamfer_loss)
        point_loss += point_lossTemp
        tangent_loss += tangent_lossTemp

        #this cannot handle -1 in lap index. need to be revisited if we want to use lap loss
        #pt_lap_loss += lap_const[idx] * laplace_loss(feat, output, self.ellipse, idx + 1)
        #print(f" tangent_loss: {tangent_loss}, point_loss: {point_loss}, blended_loss: {blended_loss}")
        #loss = 100 * pt_chamfer_loss + 0.1 * pt_edge_loss + 0.3 * pt_lap_loss
        #loss = pt_chamfer_loss + 0.1 * pt_edge_loss   #  start with only chamfer loss
        loss = blended_loss
        return [loss,point_loss,tangent_loss,diceLossVal]

    def optimizer_step(self, images, labels, df):
        self.optimizer.zero_grad()
        #loss, output1, output2, output3 = self.get_loss(images, labels)
        losses, output1 = self.get_loss(images, labels,self.ncoords,df)
        loss=losses[0]
        pointLoss=losses[1]
        tangentLoss=losses[2]
        diceLoss=losses[3]
        #write tangent loss and point losses to tensor board
        self.writer.add_scalar('pointLoss', pointLoss, self.global_step)
        self.writer.add_scalar('tangentLoss', tangentLoss, self.global_step)
        self.writer.add_scalar('diceLoss', diceLoss, self.global_step)


        loss.backward()
        self.optimizer.step()
        # if not use_cuda:
        #     return loss.item(), output1.detach().numpy(), output2.detach(
        #     ).numpy(), output3.detach().numpy()
        # else:
        #     return loss.item(), output1.detach().cpu().numpy(), output2.detach(
        #     ).cpu().numpy(), output3.detach().cpu().numpy()
        loss_ = [loss.item(), pointLoss.item(), tangentLoss.item()]
        if not use_cuda:

            return loss_, output1.detach().numpy()
        else:
            return loss_, output1.detach().cpu().numpy()
