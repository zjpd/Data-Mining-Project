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
corpus = api.load('wiki-english-20171001')  # download the corpus and return it opened as an iterable

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


# weight is a vector with shape (dim, 1)
# b is a number	
def initialize_with_zeors(dim):
	weight = np.zeros((dim, 1))
	b = 0
	return weight, b

# includes forward-propagate and backward propagate
# forward-propagate is aims to calculate the cost
# backward propagate is to get partial derivatives of W and b from the expression of cost. 
# we will first find the partial derivative of Ž.
def propagate(weight, b, x, y):
	'''
	weight -- shape: (claim_len+relevance_sentence_len, 1)
	b -- deviation
	x -- training data, shape: (claim_relevance_sentence_len, m)
	y -- label (1, m)
	
	'''
	
	m = x.shape[1]
	
	A = sigmoid(np.dot(weight.transpose(), x)+b)
	cost = -(np.sum(y * np.log(A) + (1-y) * np.log(1-A)))/m
	
	dz = A-y
	dw = (np.dot(x, dz.transpose()))/m
	db = (np.sum(dz))/m
	
	grads = {'dw': dw, 'db':db}
	
	return grads, cost


def optimize(weight, b, x, y, num_iter, learning_rate, print_cost=false):
	costs = []
	
	for i in range(num_iter):
		
		grads, cost = propagate(weight, b, x, y)
		dw = grads['dw']
		db = grads['db']
		
		weight = weight - learning_rate * dw
		b = b - learning_rate * db
		
		#every 2 times, record the cost
		if i % 2 == 0:
			costs.append(cost)
		
	params = {'weight': weight, 'b': b}
	grads = {'dw': dw, 'db': db}
	
	return params, grads, costs


def predict(weight, b, x):
	m = x.shape[1]
	y_prediction = np.zeors((1, m))
	
	A = sigmoid(np.dot(weight.transpose(), x)+b)
	
	for i in range(m):
		if A[0, 1] > 0.5:
			y_prediction[0,i] = 1
		else:
			y_prediction[0,i] = 0
	
	return y_prediction


def logistic_model(x_train, y_train, x_test, y_test, learning_rate=0.1, num_iter = 200):
	dim = x_train.shape[0]
	weight, b = initialize_with_zeors(dim)
	
	params, grads, costs = optimize(weight, b, x_train, y_train, num_iter, learning_rate, False)
	weight = params['weight']
	b = params['b']
	
	prediction_train = predict(weight, b, x_test)
	prediction_test = predict(weight, b, x_train)
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
