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
	'word': frequency
'''
def get_assigned_text(index):
	dir_path = 'D://document//UCL//Data Mining//data//wiki-pages//'
	text = []
	word = []
	doc_len = 0
	
	with open(dir_path+index, 'r') as file:
		lines = file.readlines()
		for line in lines:
			text.append(json.loads(line)['text'])
	
	for line in text:
		word.append(Counter(re.findall(r'\w+', line)))
		doc_len += len(re.findall(r'\w+', line))
	
	return word, doc_len

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


def jelinek_smoothing(query, document, doc_len):
	


def dirichilet_smoothing(query, document, doc_len):






	


	