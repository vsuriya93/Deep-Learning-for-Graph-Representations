import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.animation as animation


def load_data():
	f = open('karate/input')
	f=f.readlines()
	features = np.eye(34)
	adj = np.zeros((34,34))
	for row in f:
		row_map = map(int,row.split(' '))
		for c in row_map[1:]:
			adj[row_map[0]-1][c-1]=1
			adj[c-1][row_map[0]-1]=1
        f=open('karate/labels')
        f=f.readlines()
        clas={}
        for row in f:
        	temp=row.strip().split('-')
        	for x in map(int,temp[0].split(',')):
	        	clas[x]=int(temp[1])
	y=tf.one_hot(np.array(clas.values()),4).eval(session=tf.Session())
	return adj,features,y,clas.values()

def glorot(shape):
	init_range = np.sqrt(6.0/(shape[0]+shape[1]))
	initial = tf.random_uniform(shape,minval=-init_range,maxval=init_range,dtype=tf.float32)
	return tf.Variable(initial)

adj,features,y,q = load_data()
index = [0,2,4,8] # index of 1 labeled node per community
mask = np.array(np.zeros(y.shape[0]),dtype=bool)
mask[index]=True

new_y = np.zeros(y.shape)
new_y[index]=y[index]

adj = adj +  np.eye(adj.shape[0])
D = np.diag(np.power(adj.sum(1),-.5))
D[np.isinf(D)]=0.0
adj = np.dot(np.dot(D.T,adj),D)


support = tf.placeholder(tf.float32,shape=[None,None])
x = tf.placeholder(tf.float32,shape=[None,None])
labels = tf.placeholder(tf.float32,shape=[None,y.shape[1]])
act = tf.nn.tanh


sess=tf.Session()

w_1 = glorot([34,2])
h_1 = act(tf.matmul(tf.matmul(support,x),w_1))
#w_2 = glorot([2,2])
#h_2 = act(tf.matmul(tf.matmul(support,h_1),w_2))
#w_3 = glorot([2,2])
#h_3 = act(tf.matmul(tf.matmul(support,h_2),w_3))
w_3 = glorot([2,4])
h_3 = act(tf.matmul(tf.matmul(support,h_1),w_3))
#w_5 = glorot([2,4])
#h_5 = act(tf.matmul(tf.matmul(support,h_4),w_5))

predict = tf.nn.softmax(h_3)

feed_dict = { support: adj, x:features,labels:new_y}

def loss(pred,y,mask):
	loss = tf.nn.softmax_cross_entropy_with_logits(pred,y)
	mask = tf.cast(mask,dtype=tf.float32)
	mask = mask / tf.reduce_mean(mask)
	loss = loss * mask
	return tf.reduce_mean(loss)

def accuracy(pred,y,mask):
	acc=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
	acc = tf.cast(acc,tf.float32)
	mask = tf.cast(mask,dtype=tf.float32)
	mask = mask /tf.reduce_mean(mask)
	acc = acc * mask
	return tf.reduce_mean(acc)

loss_=0

for weight in [w_1,w_3]:
	loss_ = loss_ + .001 * tf.nn.l2_loss(weight)

loss_ = loss_ + loss(predict,labels,mask)
acc_ = accuracy(predict,labels,mask)
opt = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss_)
sess.run(tf.initialize_all_variables())

fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax1.set_xlim([-1.5,1.5])
ax1.set_ylim([-1.5,1.5])

ax2 = fig.add_subplot(1,2,2)
ax2.set_xlim([-1.5,1.5])
ax2.set_ylim([-1.5,1.5])

out = sess.run([opt,loss_],feed_dict=feed_dict)
acc = sess.run([acc_,h_1],feed_dict=feed_dict)
q1=acc[1]
new_lb = tf.identity(predict)
q2 = np.argmax(new_lb.eval(session=sess,feed_dict=feed_dict),1)
q0 = tf.one_hot(np.array(q),3).eval(session=tf.Session())
q3 = tf.one_hot(np.array(q2),3).eval(session=tf.Session())
#print q2

scat1 = ax1.scatter(q1[:,0],q1[:,1],c=q0,s=50) 
scat2 = ax2.scatter(q1[:,0],q1[:,1],c=q3,s=50) 

from sklearn.metrics import accuracy_score

def ret(i):
	out = sess.run([opt,loss_],feed_dict=feed_dict)
	acc = sess.run([acc_,h_1],feed_dict=feed_dict)
	q1=acc[1]
	q2 = np.argmax(new_lb.eval(session=sess,feed_dict=feed_dict),1)
	accuracy_final=accuracy_score(q,q2)
	print acc[0],i,accuracy_final
	q3 = tf.one_hot(np.array(q2),3).eval(session=tf.Session())
	scat1.set_color(q0)
	scat1.set_offsets(q1)
	
	scat2.set_color(q3)
	scat2.set_offsets(q1)

	plt.title('Itertation {iter_} and Accuracy - {accuracy}'.format(iter_=i,accuracy=accuracy_final))	
		
	return scat1,scat2,

ani = animation.FuncAnimation(fig,ret,interval=10,frames=400,repeat=False)
ani.save('video1.mp4',fps=25,extra_args=['-vcodec','libx264'])
#plt.show()
#viz = sess.run([h_1,h_3],feed_dict=feed_dict)
#h_1=viz[0]
#plt=plt.figure()
"""
#pca = PCA(n_components=2)
#viz=pca.fit_transform(viz)
for i in viz:
	plt.scatter(i[:,0],i[:,1],c=q,s=100)
	plt.show()
	print i.shape
	break
"""
