import numpy as np
import argparse
from data_loader import load_data,preprocess_features
from models import MF
import pdb

cora_param = {
    'data': 'cora',
    'node_num': 2708,
    'out_dim': 7,
    'feat_num': 1433,
    'act':'leaky_relu',
    'adj_num':200,
    'lr':0.005
}
pubmed_param = {
    'data': 'pubmed',
    'node_num': 19717,
    'out_dim': 3,
    'feat_num': 500,
    'act':'tanh',
    'adj_num':200,
    'lr':0.005
}
cite_param={
    'data':'citeseer',
    'node_num':3327,
    'out_dim':6,
    'feat_num':3703,
    'act':'leaky_relu',
    'adj_num':200,
    'lr':0.005
}
blog_param={
    'data':'BlogCatalog',
    'node_num':5196,
    'feat_num':8189,
    'out_dim':6,
    'act':'tanh',
    'adj_num':2000,
    'lr':0.005
}
flickr_param={
    'data':'Flickr',
    'node_num':7575,
    'feat_num':12047,
    'out_dim':9,
    'act':'tanh',
    'adj_num':2000,
    'lr':0.005
}

data_dict={
    'cora':cora_param,
    'cite':cite_param,
    'pubmed':pubmed_param,
    'blog':blog_param,
    'flickr':flickr_param
}

def make_sampled_edge(adj,node_num,neg_num):
    hi, hj = np.zeros([node_num * node_num, 1], dtype=np.int32), np.zeros(
            [node_num * node_num, 1], dtype=np.int32)
    neg_hj = np.ones([node_num * node_num, neg_num],
                            dtype=np.float32)
    n = 0
    for i in range(node_num):
        pos_j_ids = np.where(adj[i] > 0)[1]
        neg_j_ids = np.where(adj[i] <= 0)[1]

        for pos_j in pos_j_ids:
            hi[n] = i
            hj[n] = pos_j
            neg_hj[n]=np.random.choice(neg_j_ids,neg_num,replace=False)
            n+=1
    print('All the edge data is {}'.format(n))
    return hi[:n],hj[:n],neg_hj[:n]


def make_edge_data(adj,node_num,neg_num):
    k_data=make_sampled_edge(adj,node_num,neg_num)
    return [k_data]

def test2(featless=True,lr=0.005,batch_num=10,emd_unit=True,batch_size=256,data_param=cora_param):

    data = load_data(data_param['data'])
    node_num = data_param['node_num']
    out_dim = data_param['out_dim']
    feat_num = data_param['feat_num']

    neg_num=data_param['neg_num']
    adj, feas = data[:2]
    adj = adj.todense()
    # normed_adj = preprocess_adj(adj,False)
    normed_adj=adj
    edge_data=make_edge_data(adj,node_num,neg_num)[0]
    feas = preprocess_features(feas, False)

    y_train, y_val, y_test = data[2:5]
    train_mask, val_mask, test_mask = data[5:]
    accs=[]
    for _ in range(10):
        model = MF(neg_num=data_param['neg_num'],feat_less=featless, feat_num=feat_num,
        out_dim=out_dim,batch_num=batch_num,emd_act=data_param['act'],
        emd_unit=emd_unit,node_num=node_num)
        train_data = feas, y_train, train_mask
        val_data = feas, y_val, val_mask
        test_data = feas, y_test, test_mask
        res=model.fit(edge_data,train_data,val_data, test_data, early_stop_num=10,learning_rate=data_param['lr'],bz=batch_size)
        accs.append(res)
    return sum(accs)/10

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=str,default='cora')
    parser.add_argument('--emd_unit',type=int,default=1)
    parser.add_argument('--batch_num',type=int,default=10)
    parser.add_argument('--feat_less',type=int,default=0)
    parser.add_argument('--emd_act',type=str,default='')
    parser.add_argument('--lr',type=float,default=0.005)
    parser.add_argument('--neg_num',type=int,default=1)
    parser.add_argument('--bz',type=int,default=256)
    args=parser.parse_args()
    dp=data_dict[args.data]
    emd_unit=None
    if args.emd_unit==1:
        emd_unit=True
    else:
        emd_unit=False
    if args.feat_less==1:
        fl=True
    else:
        fl=False
    if args.emd_act is not '':
        dp['act']=args.emd_act
    dp['neg_num']=args.neg_num
    dp['lr']=args.lr
    res=test2(fl,args.lr,batch_num=args.batch_num,emd_unit=emd_unit,batch_size=args.bz,data_param=dp)


