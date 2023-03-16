import numpy as np
import torch
from skimage import io, transform
import pickle
use_cuda = torch.cuda.is_available()


class AttributeDict(dict):

    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, value):
        self[attr] = value


def convert_dict(d):
    new_d = AttributeDict()
    new_d.update(d)
    return new_d


def create_sparse_tensor(info):
    indices = torch.LongTensor(info[0])
    values = torch.FloatTensor(info[1])
    shape = torch.Size(info[2])
    sparse_tensor = torch.sparse.FloatTensor(indices.t(), values, shape)
    if use_cuda:
        sparse_tensor = sparse_tensor.cuda()
    return sparse_tensor


def construct_ellipsoid_info_pkl(pkl):
    """Ellipsoid info in numpy and tensor types"""
    coord = pkl[0]
    pool_idx = pkl[4]
    faces = pkl[5]
    lape_idx = pkl[7]

    edges = []
    for i in range(1, 4):
        adj = pkl[i][1]
        edges.append(adj[0])
    for i in range(3):
        idx = lape_idx[i].shape[0]
        np.place(lape_idx[i], lape_idx[i] == -1, idx)
    info_dict = {
        'features': coord,
        'edges': edges,
        'faces': faces,
        'pool_idx': pool_idx,
        'lape_idx': lape_idx,
        'support1': pkl[1],
        'support2': pkl[2],
        'support3': pkl[3]
    }
    return convert_dict(info_dict)


def construct_ellipsoid_info(args):
    pkl = pickle.load(open(args.info_ellipsoid, 'rb'), encoding='bytes')
    info_dict = construct_ellipsoid_info_pkl(pkl)
    if not use_cuda:
        tensor_dict = {
            'features':
                torch.from_numpy(info_dict.features),
            'edges': [torch.from_numpy(e).long() for e in info_dict.edges],
            'faces':
                info_dict.faces,
            'pool_idx':
                info_dict.pool_idx,
            'lape_idx': [
                torch.from_numpy(l).float() for l in info_dict.lape_idx
            ],
            'support1': [
                create_sparse_tensor(info) for info in info_dict.support1
            ],
            'support2': [
                create_sparse_tensor(info) for info in info_dict.support2
            ],
            'support3': [
                create_sparse_tensor(info) for info in info_dict.support3
            ]
        }
    else:
        tensor_dict = {
            'features':
                torch.from_numpy(info_dict.features).cuda(),
            'edges': [
                torch.from_numpy(e).long().cuda() for e in info_dict.edges
            ],
            'faces':
                info_dict.faces,
            'pool_idx':
                info_dict.pool_idx,
            'lape_idx': [
                torch.from_numpy(l).float().cuda() for l in info_dict.lape_idx
            ],
            'support1': [
                create_sparse_tensor(info) for info in info_dict.support1
            ],
            'support2': [
                create_sparse_tensor(info) for info in info_dict.support2
            ],
            'support3': [
                create_sparse_tensor(info) for info in info_dict.support3
            ]
        }
    return tensor_dict


def get_features(ellipse, images):
    if len(images.shape) == 4:
        batch_size = int(images.shape[0])
        return ellipse.shape.data.x.data.unsqueeze(0).expand(batch_size, -1, -1)
    else:
        return ellipse.shape.data.x


def load_image(img_path):
    img = io.imread(img_path)
    if img.shape[2] == 4:
        img[np.where(img[:, :, 3] == 0)] = 255
    img = transform.resize(img, (224, 224))
    img = img[:, :, :3].astype('float32')
    img_inp = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2)
    if use_cuda:
        img_inp = img_inp.cuda()
    return img_inp


def process_input(img_inp, y_train):
    img_inp = torch.from_numpy(img_inp).permute(2, 0, 1).float()
    y_train = torch.from_numpy(y_train)
    return img_inp, y_train


def process_output(output3):
    vert = output3.detach().cpu().numpy()[0]
    vert = np.hstack((np.full([vert.shape[0], 1], 'v'), vert))
    face = np.loadtxt('data/ellipsoid/face3.obj', dtype='|S32')
    mesh = np.vstack((vert, face))
    return mesh
