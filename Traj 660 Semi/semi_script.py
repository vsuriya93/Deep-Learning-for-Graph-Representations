import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import pickle
from IPython.core.debugger import Tracer
import scipy.sparse as sp
def load_data():
	f_1 = open('features')
	f_2 = open('adj_mat')
	adj = pickle.load(f_2)
	features = pickle.load(f_1)

	return adj,features,features

def glorot(shape):
	init_range = np.sqrt(6.0/(shape[0]+shape[1]))
	initial = tf.random_uniform(shape,minval=-init_range,maxval=init_range,dtype=tf.float32)
	return tf.Variable(initial)

adj,features,data = load_data()

features = np.eye(features.shape[0])

print features

#Tracer()()

index_1 = [0,1,2,3,5,7,10,11,12,13,78,4,6,8,9,71,267]
index_2 = [14,15,16,18,19,20,44,63,94,103,112,52,122,127,133,137,140,147,153,180,234,292,299,372,398]

mask = np.array(np.zeros(adj.shape[0]),dtype=bool)

mask[index_1]=True
mask[index_2]=True

sess=tf.Session()

y = np.zeros(adj.shape[0],dtype=np.int32)

y[index_2]=1

"""
y[index_0]=0
y[index_1]=1
y[index_2]=2
"""
y = tf.one_hot(y,2).eval(session=sess)

#adj = adj +  np.eye(adj.shape[0])
#D = np.diag(np.power(adj.sum(1),-.5))
#D[np.isinf(D)]=0.0
#adj = np.dot(np.dot(D.T,adj),D)

"""
t=features.sum(1)
r_inv = np.power(t, -1).flatten()
r_inv[np.isinf(r_inv)] = 0.
r_mat_inv = sp.diags(r_inv)
features = r_mat_inv.dot(features)

print features
"""
support = tf.placeholder(tf.float32,shape=[None,None])
x = tf.placeholder(tf.float32,shape=[None,None])
labels = tf.placeholder(tf.float32)
act = tf.nn.tanh


w_1 = glorot([660,2])
#w_1=tf.random_normal(shape=[30,2])
h_1 = act(tf.matmul(tf.matmul(support,x),w_1))
w_2 = glorot([2,2])
h_2 = act(tf.matmul(tf.matmul(support,h_1),w_2))

predict = tf.nn.softmax(h_2)

feed_dict = { support: adj, x:features,labels:y}

def loss(pred,y,mask):
	loss = tf.nn.softmax_cross_entropy_with_logits(pred,y)
	mask = tf.cast(mask,dtype=tf.float32)
	mask_fact = mask / tf.reduce_mean(mask)
	loss = loss * mask_fact
	return tf.reduce_mean(loss)

def accuracy(pred,y,mask):
	acc=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
	acc = tf.cast(acc,tf.float32)
	mask = tf.cast(mask,dtype=tf.float32)
	mask_fact = mask /tf.reduce_mean(mask)
	acc = acc * mask_fact
	return tf.reduce_mean(acc)

loss_=0


for weight in [w_1]:
	loss_ = loss_ + .01 * tf.nn.l2_loss(weight)

loss_ = loss_ + loss(predict,labels,mask)
acc_ = accuracy(predict,labels,mask)

opt = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss_)

sess.run(tf.initialize_all_variables())

for i in xrange(100):
	out = sess.run([opt,loss_,h_1],feed_dict=feed_dict)
	acc = sess.run([acc_,h_2],feed_dict=feed_dict)
	print out[1],acc[0]
	#raw_input()

q = np.argmax(sess.run(predict,feed_dict=feed_dict),1)
a = sess.run(acc_,feed_dict=feed_dict)

print "\n",q,a,np.argmax(y,1),"\n"
viz = sess.run(h_1,feed_dict=feed_dict)
#plt.scatter(viz[:,0],viz[:,1],s=100)

from sklearn.cluster import SpectralClustering as SC
clf=SC(n_clusters=2)
output = clf.fit_predict(viz)
plt.scatter(viz[:,0],viz[:,1],c=output,s=75)
plt.show()
d1=data[output==0]
d2=data[output==1]

import matplotlib.image as mpimg
im = mpimg.imread('scene.jpg')
plt.imshow(im)

for row in d1:
	row=np.reshape(row,(15,2))
	plt.scatter(row[:,0],row[:,1],c='r')
plt.show()

plt.imshow(im)
for row in d2:
	row=np.reshape(row,(15,2))
	plt.scatter(row[:,0],row[:,1],c='b')

plt.show()
