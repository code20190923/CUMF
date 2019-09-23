import tensorflow as tf 
import numpy as np 


def glorot(shape,name=None):
    rng=np.sqrt(6/(shape[0]+shape[1]))
    initial=tf.random_uniform(shape,minval=-rng,maxval=rng)
    return tf.Variable(initial,name=name)

def get_acts(name):
    if name=='relu':
        return tf.nn.relu
    elif name=='leaky_relu':
        return tf.nn.leaky_relu 
    elif name=='sigmoid':
        return tf.nn.sigmoid 
    elif name=='tanh':
        return tf.nn.tanh 
    elif name=='x':
        return lambda x:x

def masked_softmax_cross_entropy(preds, labels, mask, name='loss'):
    with tf.variable_scope(name):
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds,
                                                       labels=labels)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        loss *= mask
        return tf.reduce_mean(loss)

def masked_accuracy(preds, labels, mask=None):
    assert mask is not None, 'Mask can not be None'
    ind = np.where(mask)[0]
    preds = preds[ind]
    labels = labels[ind]
    correct_preds = np.equal(np.argmax(preds, 1), np.argmax(labels, 1))
    return np.mean(correct_preds)