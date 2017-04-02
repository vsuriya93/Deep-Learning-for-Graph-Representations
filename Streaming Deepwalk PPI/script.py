#Deepwalk - Suriya - 22/07/16
from matplotlib import pyplot as plt
from collections import defaultdict
from gensim.models import Word2Vec
from plot import plot_output
import random
import numpy as np
import os

number_of_walks_per_node=10
rand=random.Random(0)
filename="combined.csv"
graph=defaultdict(list)
embedding_size=2
undirected=True
path=os.getcwd()+'/images/'
color=['b' if x<9476 else 'r' for x in range(len(graph.keys()))]
stream_buffer=100

def random_walk(graph,start,rand=random.Random(0),length=40,restart=0):
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

def read_edge_list(file_name,graph,undirected=True):
    filename=open(file_name)
    for row in filename:
        row_processed=map(int,row.strip().split(","))
        #graph[row_processed[0]].append(row_processed[1])
        graph[row_processed[0]]
        if undirected:
            #graph[row_processed[1]].append(row_processed[0])
            graph[row_processed[1]]
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
    model=Word2Vec(data,min_count=0,size=2)
    if save:
        model.save_word2vec_format('output')
    return model

def build_streaming_edge_data(edge,graph,number_of_walks_per_node,rand=random.Random(0),restart=0):
    data=[]
    nodes=edge
    for _ in xrange(number_of_walks_per_node):
        rand.shuffle(nodes)
        for vertex in nodes:
            walk=random_walk(graph=graph,start=vertex,rand=rand,restart=restart)
            data.append(walk)
    return data

def create_image(array,color,filename):
    plt.scatter(array[:,0],array[:,1],c=color)
    plt.savefig(filename)

read_edge_list(filename, graph)
#print "finished building the graph",len(graph.keys())
#data=build_data_matrix(graph,number_of_walks_per_node,rand=rand)
#print "finished building the data matrix now going for training",len(data),len(data[0])
#build_word2vec_model(data,embedding_size=embedding_size,save=True)
model=Word2Vec(min_count=0,size=2)
model.build_vocab([graph.keys()])
filename=open(filename,'r')
count=0
temp=[]
times=0
for row in filename:
    row_processed=map(int,row.strip().split(","))
    x,y=row_processed[0],row_processed[1]
    if y not in graph[x]:
        graph[x].append(y)
    if undirected and x not in graph[y]:
        graph[y].append(x)
    if count%stream_buffer!=0:
        #data.append(build_streaming_edge_data(row_processed,graph,number_of_walks_per_node))
        temp.append(x)
        temp.append(y)
    else:
        data=build_streaming_edge_data(temp,graph,number_of_walks_per_node)
        model.train(data)
        temp=[]
    count+=1
model.save_word2vec_format('output')
#print "training over now plotting"
plot_output('output')
