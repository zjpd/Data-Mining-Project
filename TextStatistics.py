import json
import random
import re
import os
import sys
import re
import io
import gc
import numpy as np
from tqdm import tqdm
from collections import Counter
from nltk import word_tokenize
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt

#Get the documents from the path
#return:	id -- the id of each document
#			return_lines -- all of the information in that document
def get_assigned_text(index, label):
	dir_path = 'D://document//UCL//Data Mining//data//wiki-pages//'
	id = []
	return_lines = []
	
	with open(dir_path+index, 'r') as file:
		lines = file.readlines()
		for line in lines:
			tmpstr = json.loads(line)
			if tmpstr['id'] == '':
				continue
			id.append(tmpstr['id'].lower())
			return_lines.append(tmpstr[label])
	
	return id, return_lines



#plot the term frequencies into a line chart
#fre_list -- the term frequency list
#count -- how many terms would like to plot
def draw_frequency(fre_list, count):
	x = []
	y = []
	for i in range(count):
		x.append(fre_list[i][0])
		y.append(fre_list[i][1])
	
	plt.title('Term Frequency')
	plt.xlabel('Term')
	plt.ylabel('Frequency')
	plt.xticks(rotation=45)
	plt.plot(x, y)
	plt.show()
	

def write_to_csv(cnt, index):
	output_path = 'D://document//UCL//Data Mining//results'
	with io.open(output_path+'//'+str(index)+'.csv', 'w', encoding='utf8') as file:
		writer = csv.writer(file)
		for k,v in cnt.items():
			writer.writerow([k, v])

#plot the term frequencies as the zipf's law
#count -- how many terms would like to plot
def zip_law(count):
	x = []
	for i in range(len(tmp)):
		x.append(tmp[i][1])

	total = np.sum(np.array(x))
	words_num = len(tmp)

	x = tmp
	prob = []
	r = []
	for i in range(len(x)):
		probability = tmp[i][1]/total
		rp = (i+1) * probability
		prob.append(probability)
		r.append(rp)

	with io.open(output_path+'//zip_law.csv', 'w', encoding='utf8') as file:
		writer = csv.writer(file)
		for i in range(len(x)):
			writer.writerow([tmp[i][0], tmp[i][1], i, prob[i], r[i]])

	x_plt = range(count)
	y_plt = prob[:count]
	plt.title('Zip law')
	plt.xlabel('Rank')
	plt.ylabel('Probability')
	plt.plot(x_plt, y_plt)
	plt.show()


def term_frequency():
	dir_path = 'D://document//UCL//Data Mining//data//wiki-pages'
	output_path = 'D://document//UCL//Data Mining//results'
	files_name = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
	
	tmpstr = ""
	for i in range(len(files_name)):	
		id, doc = get_assigned_text(files_name[i],'text')
		doc = str(doc)
		doc = doc.replace('[','')
		doc = doc.replace(']','')
		tmpstr += doc.lower()
		print(files_name[i]+' done')

	words = re.findall(r'\w+', tmpstr)
	cnt = Counter(words)
	write_to_csv(cnt, 'frequency')
	tmp = cnt.most_common()
	draw_frequency(tmp, 100)