from matplotlib import pyplot as plt
from sklearn.cluster import SpectralClustering as SC
import os
import numpy as np
import pickle
file_name='output'

def get_embeddings(file_name):
	f=open(file_name)
	s=f.readlines()
	s=[x.strip() for x in s]
	s=[row.split(" ") for row in s]
	label=[]
	data,t=[],[]
	for row in s[1:]:
		label.append(int(row[0]))
		t=[]
		for element in row[1:]:
			t.append(float(element))
		data.append(t)
	print np.array(data).shape,file_name
	return label,np.array(data)

def plot_output(file_name,count=None,image=None):
	label,data=get_embeddings(file_name)
	plt.scatter(data[:,0],data[:,1])
	plt.show()

def build_model(file_name):
	label,data=get_embeddings(file_name)
	clf=SC(n_clusters=2)
	#temp=np.transpose(data)
	output=clf.fit_predict(data)
#	s=open("color",'w')
#	pickle.dump(output,s)
	plt.scatter(data[:,0],data[:,1],c=output)
	plt.show()
#	print "here"
	return output,np.array(label),data

def plot_traj_image(filename,data):
	output,label,_=build_model(filename)
	c=['r','b','g','c','m','y','k','#eeefff']
	for l in set(output):
		index_of_data=label[output==l]
		for element in index_of_data:
			val=data[element]
			t=len(val)
			x,y=val[0:t/2],val[t/2:]
			plt.scatter(x,y,c=c[l])
		plt.show()
