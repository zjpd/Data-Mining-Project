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
import gensim.downloader as api

wiki_path = 'D://document//UCL//Data Mining//data//wiki-pages//wiki-001.jsonl'
train_path = 'D://document//UCL//Data Mining//data'
test_id =  [137334, 111897, 89891, 181634, 219028, 108281, 204361, 54168, 105095, 18708]

data = []
with open(train_path+'//train.jsonl', 'r') as file:
	lines = file.readlines()
	count = 250
	for line in lines:
		tmp = json.loads(line)
		if count%5 == 0:
			data.append(tmp)
		
		count -= 1
		if count == 0:
			break
		

evidences = []
for line in data:
	tmp = {}
	tmpstr = str(line['evidence'])
	tmpstr = tmpstr.replace('[','')
	tmpstr = tmpstr.replace(']','')
	tmpstr = list(eval(tmpstr))
	
	tmplist = []
	n = 0
	while n < len(tmpstr):
		if n+4 > len(tmpstr):
			break
		
		tmplist.append(tmpstr[n+2: n+4])
		n += 4
	
	tmp[line['id']] = tmplist
	evidences.append(tmp)
	
del data
gc.collect()

#path = get_tmpfile("word2vec.model")
corpus = api.load('glove-wiki-gigaword-300')
model = api.load('glove-wiki-gigaword-300')
model = Word2Vec(corpus)

def getTestEvidence(test_id):
	data = []
	with open(train_path+'//shared_task_dev.jsonl', 'r') as file:
		lines = file.readlines()
		for line in lines:
			tmp = json.loads(line)
			for id in test_id:
				if tmp['id'] == id:
					data.append(tmp['evidence'])
	
	test_evi = []
	for i in range(len(data)):
		tmpdict = {}
		tmp = str(data[i])
		tmp = tmp.replace('[','')
		tmp = tmp.replace(']','')
		tmp = list(eval(tmp))
	
		tmplist = []
		n = 0
		while n < len(tmp):
			if n+4 > len(tmp):
				break
		
			tmplist.append(tmp[n+2: n+4])
			n += 4
		
		tmpdict[test_id[i]] = tmplist
		test_evi.append(tmpdict)
		
	return test_evi

	
	
test_evi = getTestEvidence(test_id)
	

def getXVector(evidences, model):
	x_vector = []
	for evi in evidences:
		for key, value in evi.items():
			claim_id, result = retrievePositive(key, value)
			result_word = []
			for line in result:
				line = str(result)
				line = line.replace('\t','')
				line = line.replace('\n','')
				result_word.append(re.findall(r'\w+', line))
		
			vector = []
			for line in result_word:
				for word in line:
					try:
						vector.append(model[word])
					except KeyError:
						print (word+' not in the vocabulary')
		
			x_vector.append(vector)
	
	return x_vector
	
x_test_p = getXVector(test_evi, model)
x_test = np.zeros((len(x_test_p), 1))
for i in range(len(x_test_p)):
	x_test[i] = np.sum(np.array(x_test_p[i]))
	
y_test = np.ones((len(x_test), 1))

texts = retrieveNegative()
neg_word = []
for line in texts:
	line = line.replace('\t','')
	line = line.replace('\n','')
	neg_word.append(re.findall(r'\w+', line))

x_test_n = []
vector = []
for line in neg_word:
	for word in line:
		try:
			vector.append(model[word])
		#print(word+"; "+str(vector[word]))
		except KeyError:
			print (word+' not in the vocabulary')
	x_test_n.append(vector)
	
for i in range(len(x_test_n)):
	x_test = np.append(x_test, np.sum(np.array(x_test_n[i])))

y_tmp = np.zeros((len(x_test_n), 1))
y_test = np.append(y_test, y_tmp)

x_test = np.reshape(x_test, (len(x_test), 1))
y_test = np.reshape(y_test, (len(y_test), 1))


'''
	get x,y train part
'''
	
x_train_p = []
for evi in evidences:
	for key, value in evi.items():
		claim_id, result = retrievePositive(key, value)
		result_word = []
		for line in result:
			line = str(result)
			line = line.replace('\t','')
			line = line.replace('\n','')
			result_word.append(re.findall(r'\w+', line))
		
		vector = []
		for line in result_word:
			for word in line:
				try:
					vector.append(model[word])
				except KeyError:
					print (word+' not in the vocabulary')
		
		x_train_p.append(vector)


x_train = np.zeros((len(x_train_p), 1))
for i in range(len(x_train_p)):
	x_train[i] = np.sum(np.array(x_train_p[i]))

y_train = np.ones((len(x_train), 1))

texts = retrieveNegative()
neg_word = []
for line in texts:
	line = line.replace('\t','')
	line = line.replace('\n','')
	neg_word.append(re.findall(r'\w+', line))

x_train_n = []
vector = []
for line in neg_word:
	for word in line:
		try:
			vector.append(model[word])
		#print(word+"; "+str(vector[word]))
		except KeyError:
			print (word+' not in the vocabulary')
	x_train_n.append(vector)
	
for i in range(len(x_train_n)):
	x_train = np.append(x_train, np.sum(np.array(x_train_n[i])))

y_tmp = np.zeros((len(x_train_n), 1))
y_train = np.append(y_train, y_tmp)
	
def retrieveNegative():
	dir_path = 'D://document//UCL//Data Mining//data//wiki-pages'
	files_name = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
	
	texts = []
	for name in files_name:
		doc = get_assigned_text(name,'lines')
		random_index = np.random.randint(0, len(doc), 10)
		for index in random_index:
			for k,v in doc[index].items():
				flag, tmp = retrieveRandSent(v)
				if flag:
					texts.append(tmp)
				else:
					continue
		
		print(name+' done!')
	
	return texts
	


'''
	return a positive sentences of the given claim that includes the words. 
	input the sentence id and sentence id in the evidence_list
	input the claim id
'''
def retrievePositive(claim_id, evidence_list):
	dir_path = 'D://document//UCL//Data Mining//data//wiki-pages'
	files_name = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
	
	line_list = []
	for name in files_name:
		doc = get_assigned_text(name,'lines')
		for line in doc:
			for id, lines in line.items():
				for evidence in evidence_list:
					if str(id) == evidence[0]:
						line_list.append(retrieveSentences(lines, evidence[1]))
		print(name+' done')
	
	return claim_id, line_list

def retrieveRandSent(tmpstr):
	tmptext = re.split(r'[0-9]{1,2}\t+', tmpstr)
	tmptext = tmptext[1:len(tmptext)-1]
	tmpindex = re.findall(r'[0-9]{1,4}\t+', tmpstr)
	for i in range(len(tmpindex)):
		tmpindex[i] = tmpindex[i][0:-1]

	tmp = getSentences(tmptext, tmpindex, 0)
	if len(tmp) == 0:
		return False, tmp
	else:
		randindex = np.random.randint(0, len(tmp), 1)
		return True, tmp[randindex[0]]

def retrieveSentences(tmpstr, index):

	tmptext = re.split(r'[0-9]{1,2}\t+', tmpstr)
	tmptext = tmptext[1:len(tmptext)-1]
	tmpindex = re.findall(r'[0-9]{1,4}\t+', tmpstr)
	for i in range(len(tmpindex)):
		tmpindex[i] = tmpindex[i][0:-1]

	tmp = getSentences(tmptext, tmpindex, 0)
	
	return tmp[index]

'''
	remove the mis matched sentences by sorting the index of each sentence
'''
def getSentences(text, index, count):
	#print(index[count], count, len(index))
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
	the method will return a dictionary contains all documents in a wiki-page and their id
	'id': lines
'''
def get_assigned_text(index, label):
	dir_path = 'D://document//UCL//Data Mining//data//wiki-pages//'
	document = []
	
	with open(dir_path+index, 'r') as file:
		lines = file.readlines()
		for line in lines:
			tmp = {}
			tmpstr = json.loads(line)
			if tmpstr['id'] == '':
				continue
			tmp[tmpstr['id']] = tmpstr[label].lower()
			document.append(tmp)
	
	return document
		



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


def optimize(weight, b, x, y, num_iter, learning_rate, print_cost=False):
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
	m = x.shape[0]
	y_prediction = np.zeros((m, 1))
	
	A = sigmoid(np.dot(weight.transpose(), x)+b)
	
	for i in range(m):
		if A[i, 0] > 0.5:
			y_prediction[i,0] = 1
		else:
			y_prediction[i,0] = 0
	
	return y_prediction


def logistic_model(x_train, y_train, x_test, y_test, learning_rate=0.1, num_iter = 200):
dim = x_train.shape[0]
weight, b = initialize_with_zeors(dim)
	
params, grads, costs = optimize(weight, b, x_train, y_train, num_iter, learning_rate, False)
weight = params['weight']
b = params['b']
	
tmp = weight[:len(x_test), :len(x_test)]

prediction_train = predict(tmp, b, x_test)
prediction_test = predict(weight, b, x_train)

accuracy_train = 1 - np.mean(np.abs(prediction_train - y_train))

	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
