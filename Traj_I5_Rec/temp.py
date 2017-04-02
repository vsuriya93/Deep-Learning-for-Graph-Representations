import os
import pickle
from matplotlib import pyplot as plt

path=os.getcwd()+'/cluster_8'
c=['r','b','g','c','m','y','k','#eeefff']
index=0
for fname in os.listdir(path):
	f=open(path+'/'+fname)
	data=pickle.load(f)
	print len(data)
	for key in data.keys():
		val=data[key]
                t=len(val)
                x,y=val[0:t/2],val[t/2:]
                plt.scatter(x,y,c=c[index])
	index+=1
plt.show()			
