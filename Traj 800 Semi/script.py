import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import pickle
from IPython.core.debugger import Tracer
import scipy.sparse as sp
def load_data():

	f_1 = open('aff_mat')
	f_2 = open('t.txt')

	aff_mat = pickle.load(f_1)
	f_2=f_2.readlines()
	label = np.array([map(int,x.strip()) for x in f_2]).flatten() - 1
	return aff_mat,aff_mat,label

def glorot(shape):
	init_range = np.sqrt(6.0/(shape[0]+shape[1]))
	initial = tf.random_uniform(shape,minval=-init_range,maxval=init_range,dtype=tf.float32)
	return tf.Variable(initial)

adj,adj1,label = load_data()

print adj.shape,len(label)
#Tracer()()

adj = adj +  np.eye(adj.shape[0])
D = np.diag(np.power(adj.sum(1),-.5))
D[np.isinf(D)]=0.0
adj = np.dot(np.dot(D.T,adj),D)

support = tf.placeholder(tf.float32,shape=[None,None])
labels = tf.placeholder(tf.float32)
act = tf.nn.tanh

sess = tf.Session()
w_1 = glorot([800,2])
#w_1=np.array([[-.5,.5],]*800,dtype=np.float32)
#w_1=tf.random_normal([800,2])

h_1 = act(tf.matmul(support,w_1))
w_2 = glorot([2,2])
h_2 = act(tf.matmul(tf.matmul(support,h_1),w_2))
w_3 = glorot([2,2])
h_3 = act(tf.matmul(tf.matmul(support,h_2),w_3))

feed_dict = { support: adj,labels:label}

sess.run(tf.initialize_all_variables())

viz = sess.run(h_1,feed_dict=feed_dict)

viz = (viz-viz.mean())/(viz.max()-viz.min())

plt.scatter(viz[:,0],viz[:,1],c=label)
plt.show()

from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_score as h

clf=KMeans(n_clusters=8)
output = clf.fit_predict(viz)

plt.scatter(viz[:,0],viz[:,1],c=output)
plt.show()

print h(label,output)

f_3=open('u.txt')
traj=[]
for row in [x.strip().split(',') for x in f_3]:
	traj.append(map(float,row))

traj=np.array(traj)

color=['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
for x in xrange(len(traj)):
	mid_pt=len(traj[x])/2
	plt.scatter(traj[x][:mid_pt],traj[x][mid_pt:],c=color[output[x]])
plt.show()
"""
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
mod=pca.fit_transform(adj1)
plt.scatter(mod[:,0],mod[:,1],c=label)
plt.show()

clf=KMeans(n_clusters=8)
output = clf.fit_predict(mod)
plt.scatter(mod[:,0],mod[:,1],c=output)
plt.show()

print h(label,output)

color=['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
for x in xrange(len(traj)):
	mid_pt=len(traj[x])/2
	plt.scatter(traj[x][:mid_pt],traj[x][mid_pt:],c=color[output[x]])
plt.show()
"""
