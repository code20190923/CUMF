import os
import sys
import json
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
from networkx.readwrite import json_graph
import scipy.io as sio



def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def load_data(data_str):
    if data_str in ['Flickr','BlogCatalog']:
        return load_mat(data_str)
    else:
        return load_dataset(data_str)

def load_dataset(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(
        "data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder),
                                    max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def load_mat(data_str):
    data=sio.loadmat('data/'+data_str+'.mat')
    feats=data['Attributes']
    adj=data['Network']
    label=np.reshape(data['Label'],(-1,))-1
    n_num=np.max(label)+1
    label=np.eye(n_num)[label]
    np.random.seed(9102)
    rng=np.random.get_state()
    np.random.set_state(rng)

    L=feats.shape[0]
    ids=list(range(L))

    val_ids=set(np.random.choice(ids,int(L*0.3),replace=False))
    train_L=int(len(val_ids)*(1/3))
    test_ids=sorted(list(set(ids)-val_ids))
    train_ids=set(np.random.choice(val_ids,train_L,replace=False))
    val_ids=sorted(list(val_ids-train_ids))
    train_ids=sorted(list(train_ids))
    y_train,y_val,y_test=np.zeros([L,1],dtype=np.float32),np.zeros([L,1],dtype=np.float32),np.zeros([L,1],dtype=np.float32)
    y_train[train_ids]=label[train_ids]
    y_val[val_ids]=label[val_ids]
    y_test[test_ids]=label[test_ids]
    train_mask,val_mask,test_mask=sample_mask(train_ids,L),sample_mask(val_ids,L),sample_mask(test_ids,L)
    return adj,feats,y_train,y_val,y_test,train_mask,val_mask,test_mask


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


def preprocess_features(features, sparse=True):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    # import ipdb; ipdb.set_trace()
    try:
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        features = r_mat_inv.dot(features)
    except Exception as e:
        print(e)

    if sparse:
        return sparse_to_tuple(features)
    return features.todense()


def load_ppi_data(prefix, normalize=True, load_walks=False):
    G_data = json.load(open(prefix + "-G.json"))
    G = json_graph.node_link_graph(G_data)
    if isinstance(G.nodes()[0], int):
        conversion = lambda n: int(n)
    else:
        conversion = lambda n: n

    if os.path.exists(prefix + "-feats.npy"):
        feats = np.load(prefix + "-feats.npy")
    else:
        print("No features present.. Only identity features will be used.")
        feats = None
    id_map = json.load(open(prefix + "-id_map.json"))
    id_map = {conversion(k): int(v) for k, v in id_map.items()}
    walks = []
    class_map = json.load(open(prefix + "-class_map.json"))
    if isinstance(list(class_map.values())[0], list):
        lab_conversion = lambda n: n
    else:
        lab_conversion = lambda n: int(n)

    class_map = {conversion(k): lab_conversion(v) for k, v in class_map.items()}

    ## Remove all nodes that do not have val/test annotations
    ## (necessary because of networkx weirdness with the Reddit data)
    broken_count = 0
    for node in G.nodes():
        if not 'val' in G.node[node] or not 'test' in G.node[node]:
            G.remove_node(node)
            broken_count += 1
    print(
        "Removed {:d} nodes that lacked proper annotations due to networkx versioning issues".format(
            broken_count))

    ## Make sure the graph has edge train_removed annotations
    ## (some datasets might already have this..)
    print("Loaded data.. now preprocessing..")
    for edge in G.edges():
        if (G.node[edge[0]]['val'] or G.node[edge[1]]['val'] or
                G.node[edge[0]]['test'] or G.node[edge[1]]['test']):
            G[edge[0]][edge[1]]['train_removed'] = True
        else:
            G[edge[0]][edge[1]]['train_removed'] = False

    if normalize and not feats is None:
        from sklearn.preprocessing import StandardScaler
        train_ids = np.array([id_map[n] for n in G.nodes() if
                              not G.node[n]['val'] and not G.node[n]['test']])
        train_feats = feats[train_ids]
        scaler = StandardScaler()
        scaler.fit(train_feats)
        feats = scaler.transform(feats)

    if load_walks:
        with open(prefix + "-walks.txt") as fp:
            for line in fp:
                walks.append(map(conversion, line.split()))

    return G, feats, id_map, walks, class_map


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    # return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj, norm=True, sparse=False):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj = adj + sp.eye(adj.shape[0])
    if norm:
        adj = normalize_adj(adj)
    if sparse:
        return sparse_to_tuple(adj)
    return adj.todense()

