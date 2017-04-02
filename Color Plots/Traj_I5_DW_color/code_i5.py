import os
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from fastdtw import fastdtw
from collections import defaultdict
from gensim.models import Word2Vec
from plot_i5 import plot_output,plot_traj_image
import random
import numpy as np
import pickle
from sklearn.decomposition import PCA
input_folder=os.getcwd()+'/10_minutes_clips'
num_files=5
number_of_walks_per_node=10
rand=random.Random(0)
SHOW_IMAGES=False
METRICS=False
THRESH=5
embedding_size=2
RERUN=False

graph=defaultdict(list)
data={}
data_1={}

#im=mpimg.imread('scene.jpg')
#im_x,im_y,_=im.shape

def random_walk(graph,start,rand=random.Random(0),length=25,restart=0):
    walk=[start]
    node=start
    while len(walk)<length:
        if len(graph[node])>0:
            alpha=random.random()
            if alpha >=restart:
                next_node=rand.choice(graph[node])
                walk.append(next_node)
                node=next_node
            else:
                #next_node=rand.choice(graph[start])
                walk.append(start)
                node=start
        else:
            break
    return walk

def build_data_matrix(graph,number_of_walks_per_node=10,rand=random.Random(0),restart=0):
    data=[]
    nodes=graph.keys()
    for _ in xrange(number_of_walks_per_node):
        rand.shuffle(nodes)
        for vertex in nodes:
            walk=random_walk(graph=graph,start=vertex,rand=rand,restart=restart)
            data.append(walk)
    return data

def build_word2vec_model(data,embedding_size=2,save=True):
    model=Word2Vec(data,min_count=0,size=embedding_size)
    if save:
        model.save_word2vec_format('output_i5')
    return model 

def add_edge(graph,i,j,undirected=True):
	graph[i].append(j)
	if undirected:
		graph[j].append(i)		

def euclidean_distance(x,y):
	return ((sum([value**2 for value in np.array(x)-np.array(y)]))**.5)/14

def build_graph(data,graph):
	temp=[]
	for index_i,i in enumerate(data):
		for index_j,j in enumerate(data):
			if index_i<index_j:
				distance,_=fastdtw(data[i],data[j])
				#print i,j,distance
				temp.append(distance)
				if distance <= 3000:
					add_edge(graph,i,j,undirected=True)
#	print min(temp),max(temp)

if RERUN:
	f=open("data")
	data=pickle.load(f)
	build_graph(data,graph)
        f=open("graph_data","w")
        pickle.dump(graph,f,protocol=pickle.HIGHEST_PROTOCOL)

else:
	f=open("graph_data")
	graph=pickle.load(f)
	f=open("data_1")
	data=pickle.load(f)
	
data_matrix=build_data_matrix(graph,number_of_walks_per_node,rand=rand,restart=0)
print "finished building the data matrix now going for training",len(data_matrix),len(data_matrix[0])

build_word2vec_model(data_matrix,embedding_size=embedding_size,save=True)
print "training over now plotting"

#plot_output('output_i5')
plot_traj_image('output_i51',data)
