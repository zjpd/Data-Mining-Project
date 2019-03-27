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

train_path = 'D://document//UCL//Data Mining//data'
wiki_path = 'D://document//UCL//Data Mining//data//wiki-pages//wiki-001.jsonl'
dir_path = 'D://document//UCL//Data Mining//data//wiki-pages'
claim_id = [75397, 150448, 214861, 156709, 129629, 33078, 6744, 226034, 40190, 76253]

files_name = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]

data = []
with open(train_path+'//train.jsonl', 'r') as file:
	lines = file.readlines()
	for line in lines:
		tmp = json.loads(line)
		for id in claim_id:
			if tmp['id'] == id:
				data.append(tmp)

claim = []
for line in data:
	claim.append(line['claim'])

del data
gc.collect()

query = re.findall(r'\w+', claim[0])
query = list(set(query))

'''
	index is the file name of the assigned text to read
	the method will return a dictionary contains all the words and their frequencies.
	'document': frequency
'''
def get_assigned_text(index):
	dir_path = 'D://document//UCL//Data Mining//data//wiki-pages//'
	text = []
	document = []
	doc_len = 0
	
	with open(dir_path+index, 'r') as file:
		lines = file.readlines()
		for line in lines:
			text.append(json.loads(line)['text'])
	
	for line in text:
		document.append(Counter(re.findall(r'\w+', line)))
		doc_len += len(re.findall(r'\w+', line))
	
	return document, doc_len

'''
	query is a list contains all the words in the query
	document is a list contains counters that contains words in the document and their frequenceis
	doc_len is the lenght of the document
'''
def unigram_query_likelihood_model(query, document, doc_len):
	probability = {}
	
	for word in query:
		probability[word] = 0
		for i in range(len(document)):
			for k,v in document[i].items():
				if word == k:
					probability[word] += v
	
	result = 1.0
	
	for k,v in probability.items():
		result = result * (v/doc_len)
		print (k+': '+ str(v) +' '+ str(result))
	
	return result


def laplace_smoothing(query, document, doc_len):
	probability = {}
	uni_word = set()
	
	for word in query:
		probability[word] = 0
		for i in range(len(document)):
			for k,v in document[i].items():
				uni_word.add(k)
				if word == k:
					probability[word] += v
	
	print ('uni word: '+str(len(uni_word)))
	
	for word in probability:
		probability[word] += 1
	
	result = 1.0
	
	for k,v in probability.items():
		result = result * (v/(doc_len+len(probability)))
		print (k+': '+ str(v) +' '+ str(v/(doc_len+len(uni_word))))
	
	return result


'''
	add a variable and compute the probability of word w in the collection P(w|c)
	query is a list contains all the unique word in the claim
	document is a list contains counters that is a dictionaory with words and their frequencies
	doc_len is the length of the document
'''
def jelinek_smoothing(query, document, doc_len):
	doc_prob = {}
	col_prob = {}
	var_lambda = 0.5
	
	for word in query:
		doc_prob[word] = 0
		for i in range(len(document)):
			for k,v in document[i].items():
				if word == k:
					doc_prob[word] += v
	
	col_len = 0.0
	
	dir_path = 'D://document//UCL//Data Mining//data//wiki-pages//'
	files_name = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
	
	for word in query:
		col_prob[word] = 0
	
	for name in files_name:
		word_counter = []
		with open(dir_path+name, 'r') as file:
			lines = file.readlines()
			for line in lines:
				word_counter.append(Counter(re.findall(r'\w+', json.loads(line)['text'])))
				col_len += len(re.findall(r'\w+', json.loads(line)['text']))
		
		for word in query:
			for i in range(len(word_counter)):
				for k,v in word_counter[i].items():
					if word == k:
						col_prob[word] += v
			print(word+' tf: '+str(col_prob[word]))
			
		print(name+' done, col len: '+str(col_len))
	
	result = 1.0
	for word in query:
		result *= var_lambda * (doc_prob[word]/doc_len) + (1-var_lambda) * (col_prob[word]/col_len)
	
	return result



def dirichlet_smoothing(query, document, doc_len):
	const_n = 4000000
	doc_prob = {}
	col_prob = {}
	
	for word in query:
		doc_prob[word] = 0
		for i in range(len(document)):
			for k,v in document[i].items():
				if word == k:
					doc_prob[word] += v
	
	col_len = 0.0
	
	dir_path = 'D://document//UCL//Data Mining//data//wiki-pages//'
	files_name = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
	
	for word in query:
		col_prob[word] = 0
	
	for name in files_name:
		word_counter = []
		with open(dir_path+name, 'r') as file:
			lines = file.readlines()
			for line in lines:
				word_counter.append(Counter(re.findall(r'\w+', json.loads(line)['text'])))
				col_len += len(re.findall(r'\w+', json.loads(line)['text']))
		
		for word in query:
			for i in range(len(word_counter)):
				for k,v in word_counter[i].items():
					if word == k:
						col_prob[word] += v
			print(word+' tf: '+str(col_prob[word]))
			
		print(name+' done, col len: '+str(col_len))
		print('')
		
	result = 0
	for word in query:
		tmp = math.log(((doc_len/(doc_len+const_n)) * (doc_prob[word]/doc_len)) + ((const_n/(doc_len+const_n)) * (col_prob[word]/col_len)) ,10)
		result += tmp
		print(word+': '+str(tmp))
	
	return result