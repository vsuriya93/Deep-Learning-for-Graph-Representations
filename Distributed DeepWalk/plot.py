from matplotlib import pyplot as plt

file_name='output'

def get_embeddings(file_name):
	f=open(file_name)
	s=f.readlines()
	s=[x.strip() for x in s]
	s=[row.split(" ") for row in s]
	label=[]
	x=[]
	y=[]
	for row in s[1:]:
		label.append(int(row[0]))
		x.append(float(row[1]))
		y.append(float(row[2]))
	return label,x,y

def plot_output(file_name,count=None,image=None):
	label,x,y=get_embeddings(file_name)
	for index,_ in enumerate(label):
		if index<=9476:  #positive
			c='r'
		else:           #negative
			c='b'
		plt.scatter(x[index],y[index],c=c)
	plt.show()
if __name__=="__main__":
	plot_output(file_name)
