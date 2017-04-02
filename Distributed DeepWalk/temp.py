from pyspark import SparkContext
from collections import defaultdict
from gensim.models import Word2Vec
import random

undirected=True
#file_name="youtube_input.edgelist"
file_name="inp"
graph=defaultdict(list)

sc = SparkContext("local", "Simple App")
def load_edgelist(file_name,graph):
	l=sc.textFile(file_name)
	#s=sc.parallelize(l.collect()).map(lambda row : map(int,row.strip().split('\t'))).groupByKey()
	s=l.map(lambda row : map(int,row.strip().split('\t'))).groupByKey()
	data=s.collect()
	for row in data:          
	    key=row[0]
	    for value in row[1]:
	        graph[key].append(value)
	        if undirected:
	            graph[value].append(key)
	print "\n\nLoading File Complete !!\n"
	print "Graph has ",len(graph.keys())," nodes\n\n"

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

load_edgelist(file_name,graph)
data=generate_walks(graph)
data_matrix=[]
for row in data:
	data_matrix.append(row)
model=Word2Vec(data_matrix,min_count=1,size=2)
model.save_word2vec_format("output")
