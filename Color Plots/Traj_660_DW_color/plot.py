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
	print data.shape
	clf=SC(n_clusters=2)
	#temp=np.transpose(data)
	output=clf.fit_predict(data)
	s=open("color",'w')
	pickle.dump(output,s)
	s=output
	plt.scatter(data[:,0][s==0],data[:,1][s==0],marker='+',s=45,label='Class 1',c='r')
	plt.scatter(data[:,0][s==1],data[:,1][s==1],marker='o',s=45,label='Class 2',c='b')
	plt.legend(loc='upper left')
	plt.show()
	return output,np.array(label),data

def plot_traj_image(filename,data,image):
	output,label,_=build_model(filename)
	im_x,im_y,_=image.shape
	print len(data)
	# do for output=1
	h=['r','b']
	for l in set(output):
		index_of_data=label[output==l]
		plt.imshow(image)
		for element in index_of_data:
			it=iter(data[element])
			for x in it:
				new_x= x
				new_y=next(it)
				#new_x= x*im_x
				#new_y=next(it)*im_y
				plt.scatter(new_x,new_y,c=h[l])
		plt.show()
		raw_input('Enter key to continue')
#output,label,x,y=build_model()
