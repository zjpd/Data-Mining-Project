import json
import random
import re
import os
import sys
import re
import io
import gc
import math
from tqdm import tqdm
from collections import Counter
from nltk import word_tokenize
from os import listdir
from os.path import isfile, join
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
import numpy as np

wiki_path = 'D://document//UCL//Data Mining//data//wiki-pages//wiki-001.jsonl'
train_path = 'D://document//UCL//Data Mining//data'
claim_id = [75397, 150448, 214861, 156709, 129629, 33078, 6744, 226034, 40190, 76253]

data = []
with open(train_path+'//train.jsonl', 'r') as file:
	lines = file.readlines()
	for line in lines:
		tmp = json.loads(line)
		for id in claim_id:
			if tmp['id'] == id:
				data.append(tmp)

evidence = []
for line in data:
	evidence.append(line['evidence'])

tmp = str(evidence)
tmp = tmp.replace('[','')
tmp = tmp.replace(']','')
evidence = list(eval(tmp))

n=0
token = []
while n<len(evidence):
	if n+4>len(evidence):
		break
	
	token.append(evidence[n+2:n+4])
	n += 4
	
del data
gc.collect()

train_token = {}
tmp = set()
for tmplist in token:
	tmp.add(tmplist[0])

tmp = list(tmp)
for id in tmp:
	evidencelist = []
	for tmplist in token:
		if id == tmplist[0]:
			evidencelist.append(tmplist[1])
	train_token[id] = evidencelist


dir_path = 'D://document//UCL//Data Mining//data//wiki-pages'
files_name = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]


input_sent = []
for name in files_name:
	document = get_assigned_text(name)
	for line in document:
		for k,v in line.items():
			for key, value in train_token.items():
				if k == key:
					input_sent.append(retrieveSentences(k,v, value))
	print(name+' done')

for i in range(len(input_sent)):
	for j in range(len(input_sent[i])):
		input_sent[i][j] = input_sent[i][j].replace('\n','')
		input_sent[i][j] = input_sent[i][j].replace('\t','')


def retrieveSentences(id, tmpstr, index):

	tmptext = re.split(r'[0-9]{1,2}\t+', tmpstr)
	tmptext = tmptext[1:len(tmptext)-1]
	tmpindex = re.findall(r'[0-9]{1,4}\t+', tmpstr)
	for i in range(len(tmpindex)):
		tmpindex[i] = tmpindex[i][0:-1]

	tmp = getSentences(tmptext, tmpindex, 0)
	
	sentences = []
	for item in index:
		sentences.append(tmp[item])
	
	return sentences
	
def getSentences(text, index, count):
	print(index[count], count, len(index))
	if count >= len(index)-1:
		return text
	else:
		if int(index[count]) != count:
			text[count-1] += text[count]
			text.remove(text[count])
			index.remove(index[count])
			count -= 1
		count += 1
		return getSentences(text, index, count)
		
'''
	index is the file name of the assigned text to read
	the method will return a dictionary contains all documetns in a wiki-page and their id
	'id': document
'''
def get_assigned_text(index):
	dir_path = 'D://document//UCL//Data Mining//data//wiki-pages//'
	document = []
	
	with open(dir_path+index, 'r') as file:
		lines = file.readlines()
		for line in lines:
			tmp = {}
			tmpstr = json.loads(line)
			if tmpstr['id'] == '':
				continue
			tmp[tmpstr['id']] = tmpstr['lines'].lower()
			document.append(tmp)
	
	return document
		

for line in document:
	if count >= 5:
		break
	for k,v in line.items():
		sentences.append(re.findall(r'\w+', v))
	count += 1

#path = get_tmpfile("word2vec.model")
model = Word2Vec(sentences, size=100, window=5, min_count=1, workers=4)
model.save("word2vec.model")

vector = {}
#intersection = set(claim[0]) & set(sentences)
for word in doc:
	vector[word] = model[word]
	print(word+"; "+str(vector[word]))





'''
	logistic regression part
'''


'''
	return  the sigmoid value of z
'''
def sigmoid(z):
	return 1/(1+np.exp(-z))

def read_data():

'''
	input is the input data
	label is the output y
'''
def grad_descent(input, label, numIter=150):
	m, n = np.shape(input)
	weights = np.ones(n)
	for i in range(numIter):
		dataIndex = list(range(m))
		for j in range(m):
			alpha = 4 / (1 + i + j) + 0.01 #保证多次迭代后新数据仍然有影响力
			randIndex = int(np.random.uniform(0, len(dataIndex)))
			h = sigmoid(sum(input[j] * weights))  # 数值计算
			error = label[j] - h
            weights = weights + alpha * error * input[i]
            del(dataIndex[randIndex])
	return weights

	
'''
	input: train_x is a mat datatype, each row stands for one sample
		   train_y label
		   opt optimize option
'''
def logisticRegression(train_x, train_y, opt):


	
	
	
	
	
	
	
	
	
	
	
	
	
