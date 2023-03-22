import numpy as np
import torch
from skimage import io, transform
import pickle
use_cuda = torch.cuda.is_available()
from scipy.sparse.linalg.eigen.arpack import eigsh
import networkx as nx
import scipy.sparse as sp
from nkShapeGraph import point2D

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
    points = output3.detach().cpu().numpy()[0]
    return points


def load_obj(fn, no_normal=False):
    fin = open(fn, 'r')
    lines = [line.rstrip() for line in fin]
    fin.close()

    vertices = [];
    normals = [];
    faces = [];
    for line in lines:
        if line.startswith('v '):
            vertices.append(np.float32(line.split()[1:4]))
        elif line.startswith('vn '):
            normals.append(np.float32(line.split()[1:4]))
        elif line.startswith('f '):
            faces.append(np.int32([item.split('/')[0] for item in line.split()[1:4]]))

    mesh = dict()
    mesh['faces'] = np.vstack(faces)
    mesh['vertices'] = np.vstack(vertices)

    if (not no_normal) and (len(normals) > 0):
        assert len(normals) == len(vertices), 'ERROR: #vertices != #normals'
        mesh['normals'] = np.vstack(normals)

    return mesh


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k + 1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)


def dense_cheb(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k + 1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return t_k


def unpool_face(old_faces, old_unique_edges, old_vertices):
    old_faces = np.array(old_faces)
    N = old_vertices.shape[0]
    mid_table = np.zeros((N, N), dtype=np.int32)
    new_edges = []
    new_faces = []
    for i, u in enumerate(old_unique_edges):
        mid_table[u[0], u[1]] = N + i
        mid_table[u[1], u[0]] = N + i
        new_edges.append([u[0], N + i])
        new_edges.append([N + i, u[1]])

    for i, f in enumerate(old_faces):
        f = np.sort(f)
        mid1 = mid_table[f[0], f[1]]
        mid2 = mid_table[f[0], f[2]]
        mid3 = mid_table[f[1], f[2]]

        new_faces.append([f[0], mid1, mid2])
        new_faces.append([f[1], mid1, mid3])
        new_faces.append([f[2], mid2, mid3])
        new_faces.append([mid1, mid2, mid3])

        new_edges.append([mid1, mid2])
        new_edges.append([mid2, mid3])
        new_edges.append([mid3, mid1])

    new_faces = np.array(new_faces, dtype=np.int32)
    new_edges = np.array(new_edges, dtype=np.int32)
    return new_edges, new_faces


def write_obj(path, vertices, faces):
    with open(path, 'w') as o:
        for v in vertices:
            o.write('v {} {} {}\n'.format(v[0], v[1], v[2]))
        for f in faces:
            o.write('f {} {} {}\n'.format(f[0] + 1, f[1] + 1, f[2] + 1))


def cal_lap_index(mesh_neighbor):
    lap_index = np.zeros([mesh_neighbor.shape[0], 2 + 8]).astype(np.int32)
    for i, j in enumerate(mesh_neighbor):
        lenj = len(j)
        lap_index[i][0:lenj] = j
        lap_index[i][lenj:-2] = -1
        lap_index[i][-2] = i
        lap_index[i][-1] = lenj
    return lap_index