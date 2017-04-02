from pyspark import SparkContext
from collections import defaultdict
from pyspark.mllib.feature import Word2Vec
from pyspark.mllib.feature import Word2VecModel
from pyspark.mllib.classification import SVMWithSGD
from pyspark.mllib.regression import LabeledPoint
import os,tempfile,copy
import random,time
import pickle

undirected=True
#file_name="youtube_input.edgelist"
file_name="combined.csv"
#file_name="new"
graph=defaultdict(list)

sc = SparkContext("local", "Simple App")
def load_edgelist(file_name,graph):
	l=sc.textFile(file_name)
	#s=sc.parallelize(l.collect()).map(lambda row : map(int,row.strip().split('\t'))).groupByKey()
	s=l.map(lambda row : row.strip().split(',')).groupByKey()
	data=s.collect()
	for row in data:          
	    key=row[0]
	    for value in row[1]:
	        graph[key].append(value)
	        if undirected:
	            graph[value].append(key)
#	print "Graph has ",len(graph.keys())," nodes"

def random_walk(graph,start=None,walk_length=40,rand=random.Random(0)):
	walk=[start]
	node=start
	while len(walk)<walk_length:
		next_node=rand.choice(graph[node])
		walk.append(next_node)
		node=next_node
	return walk

def generate_walks(graph,times=10,rand=random.Random(0)):
        nodes=graph.keys()
	for _ in xrange(10):
		rand.shuffle(nodes)
		for node in nodes:
			yield random_walk(graph,rand=rand,start=node)

def getEmbeddings(row,d):
	row_vector=[]
	for x in row[1]:
		row_vector+=d[x]
	return LabeledPoint(row[0][0],row_vector)
embedding_size=2
start=time.time()
load_edgelist(file_name,graph)
end=time.time()
print "Loading edgelist\t",(end-start)
data=generate_walks(graph)
data_matrix=[]
for row in data:
	data_matrix.append(row)
print "Data Matrix Created"
s=sc.parallelize(data_matrix)
print "Building Word Vectors"
start=time.time()
model=Word2Vec().setVectorSize(embedding_size).setSeed(22).setMinCount(1).fit(s)
end=time.time()
print "Word2vec\t",(end-start)
embeddings=model.getVectors()
d=defaultdict(list)
for key in embeddings:
	for x in embeddings[key]:
		d[key].append(x)

import matplotlib.pyplot as plt
for x in d:
	s=d[x]
	plt.scatter(s[0],s[1])
	#plt.annotate('%s' %str(x),(s[0],s[1]))
plt.show()

l=sc.textFile(file_name)
X=l.map(lambda row : row.strip().split(','))
t=sc.textFile("combined_lab.csv")
y=t.map(lambda row: map(int,row.strip()))
temp=y.zip(X)
data=temp.map(lambda row : getEmbeddings(row,d))
for x in [.01,.02,.03,.04,.05,.06,.07,.08,.09]:
	train,test=data.randomSplit(weights=[x,1-x])
	start=time.time()
	svm=SVMWithSGD.train(train,step=3,regType=None)
	end=time.time()
	output=test.map(lambda row : (svm.predict(row.features),row.label))
	accuracy=1.0*output.filter(lambda (x,v) : x==v).count()/test.count()
	print accuracy,x,train.count(),test.count(),(end-start)
