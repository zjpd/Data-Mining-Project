from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM, Dropout
from keras.layers import Dense, concatenate
from keras.callbacks import Callback
from keras.callbacks import CSVLogger
from keras.callbacks import TensorBoard
from keras.models import load_model

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from openpyxl import Workbook
from openpyxl.reader.excel import load_workbook

import collections
import math
import datetime
import numpy as np
import json
import re
import io
import gc
import random
import gensim.downloader as api
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
from tqdm import tqdm
from numpy import array, newaxis
from os import listdir
from os.path import isfile, join
from nltk import word_tokenize

'''
	1. get train data that includes claim in train.jsonl
	2. get the evidences of the claim and retrieve the relevant sentences
	3. label the sentences
	4. setup the model
	MAX_SEQUENCE_LENGTH = 100 # 每条新闻最大长度
	EMBEDDING_DIM = 200 # 词向量空间维度
	VALIDATION_SPLIT = 0.16 # 验证集比例
	TEST_SPLIT = 0.2 # 测试集比例
'''

train_path = 'D://document//UCL//Data Mining//data'
data = []
with open(train_path+'//train.jsonl', 'r') as file:
	lines = file.readlines()
	for line in lines:
		tmp = json.loads(line)
		if tmp['label'] == 'SUPPORTS' or tmp['label'] == 'REFUTES':
			data.append(tmp)

data = data[:50]

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
	evidences.append(tmplist)


dir_path = 'D://document//UCL//Data Mining//data//wiki-pages'
files_name = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]

sentences = []
for i in range(len(evidences)):
	tmp = []
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
		
x_tmp = []
y_train = []	#SUPPORTS -- 1  REFUTES -- 0
for i in range(len(data)):
	tmpword = re.findall(r'\w+', data[i]['claim'].lower())
	for j in range(len(sentences[i])):
		x_tmp.append(tmpword + sentences[i][j])
		if data[i]['label'] == 'SUPPORTS':
			y_train.append(1)
		else:
			y_train.append(0)




word_model = api.load('glove-wiki-gigaword-300')
#input shape: sent_len * 300
x_train = np.zeros((len(x_tmp), 10, 300))
for i in range(len(x_tmp)):
	x_tmp[i] = x_tmp[i][:10]

for i in range(len(x_tmp)):
	for j in range(len(x_tmp[i])):
		try:
			tmp = word_model[x_tmp[i][j]]
		except KeyError:
			tmp = np.zeros((300, 1))
			print(x_tmp[i][j] + ' not in the vocabulary')
		
		for k in range(len(tmp)):
			x_train[i][j][k] = tmp[k]

y_train = np.array(y_train)

model = Sequential()
model.add(LSTM(128, activation='relu', input_shape=(10, 300)))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))
model.summary()
			
model.compile(loss='mae', optimizer='adam', metrics=['accuracy','mae'])
model.fit(x_train, y_train, batch_size = 10, verbose=1, epochs=10)
model.save('lstm_model.h5')

def get_train_data():



'''
	return a positive sentences of the given claim that includes the words. 
	input the sentence id and sentence id in the evidence_list
	input the claim id
'''

def retrieveSentences(tmpstr, index):

	tmptext = re.split(r'[0-9]{1,2}\t+', tmpstr)
	tmptext = tmptext[1:len(tmptext)-1]
	tmpindex = re.findall(r'[0-9]{1,4}\t+', tmpstr)
	for i in range(len(tmpindex)):
		tmpindex[i] = tmpindex[i][0:-1]

	tmp = getSentences(tmptext, tmpindex, 0)
	
	return tmp[index]

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