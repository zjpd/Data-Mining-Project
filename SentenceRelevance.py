import json
import random
import re
import os
import sys
import re
import io
import gc
import math
import numpy as np
import gensim.downloader as api
from tqdm import tqdm
from collections import Counter
from nltk import word_tokenize
from os import listdir
from os.path import isfile, join
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
from matplotlib import pyplot as plt

wiki_path = 'D://document//UCL//Data Mining//data//wiki-pages//wiki-001.jsonl'
train_path = 'D://document//UCL//Data Mining//data'
test_id =  [137334, 111897, 89891, 181634, 219028, 108281, 204361, 54168, 105095, 18708]

global model
model = api.load('glove-wiki-gigaword-300')


#	count -- how many training data will be returned
def get_train_data(count):
	global model
	x_train = []; y_train = []
	
	count = 250
	train_path = 'D://document//UCL//Data Mining//data'
	
	data = []
	with open(train_path+'//train.jsonl', 'r') as file:
		lines = file.readlines()
		for line in lines:
			tmp = json.loads(line)
			if count%5 == 0:
				data.append(tmp)
	
			count -= 1
			if count == 0:
				break
	
	#	retrieve the evidences of each claim
	train_evidences = getEvidences(data)
	sentences = retrieveEvi(train_evidences, data)
	
	for i in range(len(sentences)):
		tmp = []
		for j in range(len(sentences[i])):
			tmp += sentences[i][j]
		sentences[i] = tmp
		
	x_train = get_word_vector(sentences, model)
	y_train = np.ones((len(x_train), 1))
	
	print('positive part finished')
	
	texts = retrieveNegative()
	neg_word = []
	for line in texts:
		line = line.replace('\t',' ')
		line = line.replace('\n',' ')
		neg_word.append(re.findall(r'\w+', line.lower()))

	neg_word = neg_word[:len(x_train)]
	x_train_n = get_word_vector(neg_word, model)
	
	x_train = np.vstack((x_train, x_train_n))

	y_tmp = np.zeros((len(x_train_n), 1))
	y_train = np.vstack((y_train, y_tmp))
	
	return x_train, y_train

	
def get_test_data():
	x_test = []
	y_test = []

	test_id =  [137334, 111897, 89891, 181634, 219028, 108281, 204361, 54168, 105095, 18708]
	data = []
	with open(train_path+'//shared_task_dev.jsonl', 'r') as file:
		lines = file.readlines()
		for line in lines:
			tmp = json.loads(line)
			for id in test_id:
				if tmp['id'] == id:
					data.append(tmp)
	
	test_evidences = getEvidences(data)
	sentences = retrieveEvi(test_evidences, data)
	
	for i in range(len(sentences)):
		tmp = []
		for j in range(len(sentences[i])):
			tmp += sentences[i][j]
		sentences[i] = tmp
		
	x_test = get_word_vector(sentences, model)
	y_test = np.ones((len(x_test), 1))
	
	print('positive part finished')
	
	texts = retrieveNegative()
	neg_word = []
	for line in texts:
		line = line.replace('\t',' ')
		line = line.replace('\n',' ')
		neg_word.append(re.findall(r'\w+', line.lower()))

	x_test_n = get_word_vector(neg_word, model)
	
	x_test = np.vstack((x_test, x_test_n))

	y_tmp = np.zeros((len(x_test_n), 1))
	y_test = np.vstack((y_test, y_tmp))
	
	return x_test, y_test

	
def get_word_vector(sentences, model):
	matrix = np.zeros((len(sentences), 300))
	for i in range(len(sentences)):
		tmplist = []
		for j in range(len(sentences[i])):
			try:
				tmp = model[sentences[i][j]]
				tmp = np.reshape(tmp, (300, 1))
			except KeyError:
				tmp = np.zeros((300, 1))
				print(sentences[i][j] + ' not in the vocabulary')
			tmplist.append(tmp)
		
		for k in range(len(tmplist)):
			for l in range(300):
				matrix[i][l] += tmplist[k][l][0]
		
	return matrix
'''
	return an evidences list.
	data -- list, contains all the information retrieved from the train.jsonl
'''
def getEvidences(data):
	evidences = []
	for line in data:	
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
		evidences.append(tmplist)
	
	return evidences

'''
	return a list contains the relevant sentences of the claim based on the evidence list
'''
def retrieveEvi(evidences, data):
	dir_path = 'D://document//UCL//Data Mining//data//wiki-pages'
	files_name = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
	
	sentences = []
	for i in range(len(evidences)):
		tmp = []
		tmp.append(data[i]['claim'].lower())
		sentences.append(tmp)
	
	for name in files_name:
		id, doc = get_assigned_text(name, 'lines')
		for i in range(len(evidences)):
			for j in range(len(evidences[i])):
				try:
					index = id.index(evidences[i][j][0])
				except ValueError:
					continue
				sentences[i].append(retrieveSentences(doc[index], evidences[i][j][1]))
			
		print (name+' done')
	
	
	for i in range(len(sentences)):
		for j in range(len(sentences[i])):
			sentences[i][j] = sentences[i][j].replace('\t',' ')
			sentences[i][j] = sentences[i][j].replace('\n',' ')
			sentences[i][j] = re.findall(r'\w+', sentences[i][j].lower())
	return sentences
	

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
			id.append(tmpstr['id'])
			return_lines.append(tmpstr[label])
	
	return id, return_lines
	
	

def retrieveNegative():
	dir_path = 'D://document//UCL//Data Mining//data//wiki-pages'
	files_name = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
	
	texts = []
	for name in files_name:
		id, doc = get_assigned_text(name,'lines')
		random_index = np.random.randint(0, len(doc), 10)
		for index in random_index:
			flag, tmp = retrieveRandSent(doc[index])
			if flag:
				texts.append(tmp)
			else:
				continue
		
	print(name+' done!')
	
	return texts
	


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
	weight = np.ones((1, dim))
	for i in range(dim):
		weight[0][i] = 0.5
	b = 0
	return weight, b

def min_max_scaler(x):
	max = np.amax(x)
	min = np.amin(x)
	return (x - max)/(max - min)

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
	
	m = x.shape[0]

	A = sigmoid(np.dot(weight, x.transpose())+b)
	deviation = 0.000001
	for i in range(len(A[0])):
		if A[0][i] == 0:
			A[0][i] += deviation
		elif A[0][i] == 1:
			A[0][i] -= deviation
	cost = -(np.sum(y.transpose() * np.log(A) + (1-y.transpose()) * np.log(1-A)))/m
	
	dz = A-y.transpose()
	dw = (np.dot(x.transpose(), dz.transpose()))/m
	db = (np.sum(dz))/m
	
	grads = {'dw': dw, 'db':db}
	
	return grads, cost


def optimize(weight, b, x, y, num_iter, learning_rate, print_cost=False):
	costs = []
	
	for i in range(num_iter):
		
		grads, cost = propagate(weight, b, x, y)
		dw = grads['dw']
		db = grads['db']
		
		weight = weight - learning_rate * dw.transpose()
		b = b - learning_rate * db
		
		#every 2 times, record the cost
		costs.append(cost)
		
	params = {'weight': weight, 'b': b}
	grads = {'dw': dw, 'db': db}
	
	return params, grads, costs


def predict(weight, b, x):
	m = x.shape[0]
	y_prediction = np.zeros((m, 1))

	A = sigmoid(np.dot(weight, x.transpose())+b)
	
	for i in range(m):
		if A[0, i] > 0.5:
			y_prediction[i,0] = 1
		else:
			y_prediction[i,0] = 0
	
	return y_prediction


def logistic_model(learning_rate, num_iter):
	x_train, y_train = get_train_data(1000)
	x_test, y_test = get_test_data()
	
	dim = x_train.shape[1]
	weight, b = initialize_with_zeors(dim)
	
	params, grads, costs = optimize(weight, b, x_train, y_train, num_iter, learning_rate, False)
	weight = params['weight']
	b = params['b']

	prediction = predict(weight, b, x_test)

	accuracy_test = 1 - np.mean(np.abs(prediction - y_test))
	print(accuracy_test)
	#for cost in costs:
		#print(cost)
	pre = precision(prediction, y_test)
	rec = recall(prediction, y_test)
	f1 = (2*pre*rec)/(pre+rec)
	
	plt.title('learning_rate = '+str(learning_rate), fontsize=16)
	plt.xlabel('Cost', fontsize=16)
	plt.plot(costs, linewidth='3', color='black')
	plt.show()
	return pre, rec, f1
	
	
def precision(y_pred, y):
	tp = 0
	fp = 0
	for i in range(len(y_pred)):
		if y_pred[i] == 0 and y_pred[i] == y[i]:
			tp += 1
		elif y_pred[i] == 0 and y_pred[i] != y[i]:
			fp += 1
	return tp/(tp+fp)

def recall(y_pred, y):
	tp = 0
	fn = 0
	for i in range(len(y_pred)):
		if y[i] == 0 and y[i] != y_pred[i]:
			fn += 1
		elif y[i] == 0 and y[i] == y_pred[i]:
			tp += 1
	return tp/(tp+fn)
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
