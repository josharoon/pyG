import torch
import torch.optim as optim
from torch_geometric.utils import get_laplacian
from .losses import *
from .utils import *
from .chamfer import nn_distance_function as chamfer_dist
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

    def get_loss(self, img_inp, labels,ncoords):
        if type(img_inp) != list:
            batch = len(img_inp.shape) == 4
            inputs = get_features(self.ellipse, img_inp) #return data from ellipse
            outputs = self.network(img_inp)

        else:
            batch = len(img_inp[0].shape) == 4
            inputs = get_features(self.ellipse, img_inp[0]) #return data from ellipse
            outputs = self.network(img_inp[0], img_inp[1])
        if batch:
            loss = 0
            loss1 = 0
            loss2 = 0
            for idx, (input, label) in enumerate(zip(inputs, labels)):
                #output = [out[idx] for out in outputs]
                output=[outputs] #we only have one output at the moment
                loss += self._get_loss(input, output, label,ncoords)[0]
                loss1 += self._get_loss(input, output, label,ncoords)[1]
                loss2 += self._get_loss(input, output, label,ncoords)[2]
            loss /= len(inputs)
            loss1 /= len(inputs)
            loss2 /= len(inputs)
        else:
            loss = self._get_loss(inputs, outputs, labels,ncoords)
            loss1 = self._get_loss(inputs, outputs, labels,ncoords)[1]
            loss2 = self._get_loss(inputs, outputs, labels,ncoords)[2]
        #return loss, outputs[0], outputs[2], outputs[4]
        return [loss,loss1,loss2], outputs[0]

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

    def _get_loss_pt(self, inputs, outputs, labels, nCoords):
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

        pt_chamfer_loss = 0.
        pt_edge_loss = 0.
        pt_lap_loss = 0.
        pt_match_loss = 0.
        point_loss = 0.
        tangent_loss = 0.
        lap_const = [0.2, 1., 1.]
        idx=0 #we only have block 1 at the moment
        # for idx, (output, feat) in enumerate(
        #         zip([outputs[0], outputs[2], outputs[4]],
        #             [inputs, outputs[1], outputs[3]])):

        #for idx, (output, feat) in enumerate(outputs,inputs):

        output=outputs[0][0]
        feat=inputs[:,:2]
        #dist1, dist2, _, _ = chamfer_dist(output, labels[:, :nCoords])
        #pt_chamfer_loss += torch.mean(dist1) + torch.mean(dist2)
        #pt_edge_loss += edge_loss_pt(output, labels, self.ellipse,idx + 1)
        point_lossTemp =pointMatchLoss(output[:, :2], labels[:, :2],2)
        tangent_lossTemp = pointMatchLoss(output[:, 2:], labels[:, 2:],4)
        pt_match_loss += (point_lossTemp + tangent_lossTemp)
        point_loss += point_lossTemp
        tangent_loss += tangent_lossTemp

        #this cannot handle -1 in lap index. need to be revisited if we want to use lap loss
        #pt_lap_loss += lap_const[idx] * laplace_loss(feat, output, self.ellipse, idx + 1)
        #print(f" tangent_loss: {tangent_loss}, point_loss: {point_loss}, pt_match_loss: {pt_match_loss}")
        #loss = 100 * pt_chamfer_loss + 0.1 * pt_edge_loss + 0.3 * pt_lap_loss
        #loss = pt_chamfer_loss + 0.1 * pt_edge_loss   #  start with only chamfer loss
        loss = pt_match_loss
        return [loss,point_loss,tangent_loss]

    def optimizer_step(self, images, labels):
        self.optimizer.zero_grad()
        #loss, output1, output2, output3 = self.get_loss(images, labels)
        losses, output1 = self.get_loss(images, labels,self.ncoords)
        loss=losses[0]
        pointLoss=losses[1]
        tangentLoss=losses[2]
        #write tangent loss and point losses to tensor board
        self.writer.add_scalar('pointLoss', pointLoss, self.global_step)
        self.writer.add_scalar('tangentLoss', tangentLoss, self.global_step)

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
