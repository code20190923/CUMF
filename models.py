import tensorflow as tf
import pdb
from common import *
class Model(object):
    def __init__(self,node_num,feat_num,feat_less,out_dim):
        self.sess=tf.Session()
        self.node_num=node_num
        self.feat_num=feat_num
        self.out_dim=out_dim
        self.feat_less=feat_less
        self.loss=0

    def _build_model(self):
        raise NotImplementedError

    def _init_vars(self, vars):
        for var in vars:
            self.sess.run(var.initializer)

    def _fit_init(self,learning_rate):
        raise NotImplementedError

    def _tf_init(self):
        self.sess.run(tf.global_variables_initializer())

    def _pack_fit_fed(self,data):
        raise NotImplementedError

    def _fit(self, edge_data,train_data, val_data,test_data, max_epoch,early_stop_num,bz):
        raise NotImplementedError

    def shuffle(self,*args):
        rst=np.random.get_state()
        for x in args:
            np.random.set_state(rst)
            np.random.shuffle(x)

    def fit(self, edge_data,train_data, val_data,test_data,
            max_epoch=1000,
            early_stop_num=10,
            learning_rate=0.001,bz=256):
        self._fit_init(learning_rate)

        return self._fit(edge_data,train_data, val_data,test_data, max_epoch, early_stop_num,bz)


class MF(Model):
    def __init__(self,node_num=None,feat_num=None,feat_less=False,
                out_dim=None,batch_num=10,emd_act='leaky_relu',emd_unit=True,neg_num=1,out_act=['relu','x'],out_hidden=[128]):
        super(MF,self).__init__(node_num,feat_num,feat_less,out_dim)
        self.emd_act=get_acts(emd_act)
        self.emd_sz=int(self.feat_num*0.1)
        self.emd_unit=emd_unit
        self.neg_num=neg_num
        self.batch_num=batch_num
        self.out_act=out_act
        self.out_hidden=out_hidden
        self._build_model()
        self._tf_init()

    def _placeholder_init(self):
        self.inputs = tf.placeholder(tf.float32, [None, self.feat_num])
        self.ground_truth = tf.placeholder(tf.float32, [None, self.out_dim])
        self.masks = tf.placeholder(tf.int32, None, name='labels_mask')
        self.xi_id=tf.placeholder(tf.int32,[None,1])
        self.xj_id=tf.placeholder(tf.int32,[None,1])
        self.neg_xj_id=tf.placeholder(tf.int32,[None,self.neg_num])
        self.keep_prob=tf.placeholder(tf.float32)
    def _weight_init(self):
        self.w=glorot([self.feat_num,self.emd_sz])
        self.b=tf.Variable(tf.zeros([1])+0.1)
        self.out_w=[glorot([self.emd_sz,self.out_hidden[0]])]
        for i in range(1,len(self.out_hidden)):
            self.out_w.append(glorot([self.out_hidden[i-1],self.out_hidden[i]]))
        self.out_w.append(glorot([self.out_hidden[-1],self.out_dim]))

    def _struct_pred(self):
        self.hi=tf.nn.embedding_lookup(self.logits,self.xi_id)
        self.hi=tf.squeeze(self.hi)
        self.hj=tf.nn.embedding_lookup(self.logits,self.xj_id)
        self.hj=tf.squeeze(self.hj)
        self.neg_hj=tf.nn.embedding_lookup(self.logits,self.neg_xj_id)
        self.neg_hj=tf.squeeze(self.neg_hj)
        neg_hj=-1*self.neg_hj
        tmp=tf.expand_dims(self.hi,1)
        neg_p=tf.log_sigmoid(tf.reduce_sum(tf.multiply(tmp,neg_hj),-1))
        neg_part=-1*tf.reduce_mean(neg_p,-1)

        struct_p=tf.multiply(self.hi,self.hj)
        struct_p=tf.nn.sigmoid(tf.reduce_sum(struct_p,-1,keepdims=True))
        pos_part=-1*tf.log(tf.reshape(struct_p,[-1,1]))
        self.struct_loss=tf.reduce_mean(pos_part+neg_part)

    def _label_pred(self):
        out=self.logits
        for w,act in zip(self.out_w,self.out_act):
            out=tf.nn.dropout(out,self.keep_prob)
            out=get_acts(act)(tf.matmul(out,w))
        self.label_pred=out
        self.label_loss=masked_softmax_cross_entropy(self.label_pred,self.ground_truth,self.masks)
        l2_loss=0
        for w in self.out_w:
            l2_loss+=tf.nn.l2_loss(w)
        self.label_loss+=0.0002*l2_loss
    def _inference(self):
        if self.feat_less:
            self.pre_logits=glorot([self.node_num,self.emd_sz])
        else:
            self.pre_logits=self.emd_act(tf.matmul(self.inputs,self.w)+self.b)
        if self.emd_unit:
            mode=tf.sqrt(tf.reduce_sum(tf.square(self.pre_logits),-1,keepdims=True))
            self.logits=self.pre_logits/mode
        else:
            self.logits=self.pre_logits
        self._struct_pred()
        self._label_pred()

    def _build_model(self):
        self._placeholder_init()
        self._weight_init()
        self._inference()

    def get_batch(self,data,bz=1024):
        def make(batch_id,length,data):
            if batch_id == length:
                s=batch_id*bz
                return data[s:]
            elif batch_id>length:
                k=np.random.randint(0,length)
                s,e=k*bz,(k+1)*bz
                return data[s:e]
            else:
                s,e=batch_id*bz,(batch_id+1)*bz
                return data[s:e]

        N=0
        for d in data:
            t=len(d)//bz
            if t>N:
                N=t
        N+=1
        for i in range(N):
            lst=[]
            for d in data:
                n=len(d)//bz
                lst.append(make(i,n,d))
            if lst[0][-1].shape[0]==0:
                return
            yield lst

    def _pack_fit_fed(self, data):
        inputs, labels, masks = data
        return {
                self.inputs: inputs,
                self.ground_truth: labels,
                self.masks: masks,
            }

    def _fit_init(self,learning_rate):
        self.label_optimizer=tf.train.AdamOptimizer(learning_rate)
        self.label_opt=self.label_optimizer.minimize(self.label_loss)
        self.struct_optimizer=tf.train.AdamOptimizer(learning_rate)
        self.struct_opt=self.struct_optimizer.minimize(self.struct_loss)

        self._init_vars(self.label_optimizer.variables())
        self._init_vars(self.struct_optimizer.variables())

    def _pack_edge_data(self,data):
        xi,xj,neg=data
        return {
            self.xi_id:xi,
            self.xj_id:xj,
            self.neg_xj_id:np.reshape(neg,[-1,self.neg_num])
        }

    def _fit(self,edge_data,train_data,val_data,test_data,max_epoch,early_stop_num,bz):
        best,stop_step=0,0
        con_test_score=0
        cnt=0
        train_fed=self._pack_fit_fed(train_data)
        train_fed.update({self.keep_prob:0.5})
        val_fed=self._pack_fit_fed(val_data)
        val_fed.update({self.keep_prob:1.})
        test_fed=self._pack_fit_fed(test_data)
        test_fed.update({self.keep_prob:1.})

        for k in range(max_epoch):

            self.shuffle(*edge_data)
            for batch in self.get_batch(edge_data,bz):
                cnt+=1
                fed=self._pack_edge_data(batch)
                fed.update({self.inputs:train_data[0]})
                struct_loss=self.sess.run(self.struct_loss,fed)
                self.sess.run(self.struct_opt,fed)
                if cnt%self.batch_num==0:
                    print('struct loss is {}'.format(struct_loss))
                    _,label_loss,label_pred=self.sess.run([self.label_opt,self.label_loss,self.label_pred],train_fed)
                    val_loss,val_pred=self.sess.run([self.label_loss,self.label_pred],val_fed)
                    test_pred=self.sess.run(self.label_pred,test_fed)
                    train_acc=masked_accuracy(label_pred,train_data[1],train_data[2])
                    val_acc=masked_accuracy(val_pred,val_data[1],val_data[2])
                    test_acc=masked_accuracy(test_pred,test_data[1],test_data[2])
                    if val_acc > best:
                        best=val_acc
                        con_test_score=test_acc
                        stop_step=0
                    else:
                        stop_step+=1
                    print('epoch {}, train loss is {}, accurayc is {}, val loss is {}, accuray is {}'.format(k,
                    label_loss,train_acc,val_loss,val_acc))
                    print('test_score is {}'.format(test_acc))
                    if stop_step>=early_stop_num:
                        print('Best val score is {}, test score is {}'.format(best,con_test_score))
                        return con_test_score






