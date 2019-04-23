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

claim = []
for line in data:
	claim.append(list(set(re.findall(r'\w+', line['claim'].lower()))))

del data
gc.collect()


doc_len = 0
for k,v in document[0].items():
	doc = re.findall(r'\w+', v)
	doc_len = len(doc)
	doc = dict(Counter(doc))

def test_laplace():
	train_path = 'D://document//UCL//Data Mining//data'
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
		claim.append(list(set(re.findall(r'\w+', line['claim'].lower()))))

	del data
	gc.collect()
	
	results = []
	
	for cl in claim:
		prob_result = {}
		
		for name in files_name:
			document, doc_len = get_assigned_text(name)
			for line in document:
				for k,v in line.items():
					doc_len = len(re.findall(r'\w+', v))
					doc = dict(Counter(re.findall(r'\w+', v)))
					result = laplace_smoothing(cl, doc, doc_len)
					prob_result[k] = result
			print(name+' done!')
		
		prob_result = sorted((value,key) for (key,value) in prob_result.items())
		print(prob_result[-6:])
		results.append(prob_result[-20:])
		
	return results

testtmp = {}
for i in range(20):
	testtmp[i] = i
testtmp = sorted((value,key) for (key,value) in testtmp.items())
	
'''
	index is the file name of the assigned text to read
	the method will return a dictionary contains all documetns in a wiki-page and their id
	'id': document
'''
def get_assigned_text(index):
	dir_path = 'D://document//UCL//Data Mining//data//wiki-pages//'
	document = []
	doc_len = 0
	
	with open(dir_path+index, 'r') as file:
		lines = file.readlines()
		for line in lines:
			doc_len += 1
			tmp = {}
			tmpstr = json.loads(line)
			if tmpstr['id'] == '':
				continue
			tmp[tmpstr['id']] = tmpstr['text'].lower()
			document.append(tmp)
			#doc_len += 1
	
	return document, doc_len

'''
	query is a list contains all the words in the query
	document is a dict that contains words in the document and their frequenceis
	doc_len is the lenght of the document
'''
def unigram_query_likelihood_model(query, document, doc_len):
	probability = {}
	
	for word in query:
		probability[word] = 0
		for k,v in document.items():
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
		for k,v in document.items():
			uni_word.add(k)
			if word == k:
				probability[word] += v
	
	#print ('uni word: '+str(len(uni_word)))
	
	for word in probability:
		probability[word] += 1
	
	result = 1.0
	
	for k,v in probability.items():
		result = result * (v/(doc_len+len(probability)))
		#print (k+': '+ str(v) +' '+ str(v/(doc_len+len(uni_word))))
	
	return result


'''
	add a variable and compute the probability of word w in the collection P(w|c)
	query is a list contains all the unique word in the claim
	document is a dictionaory with words and their frequencies
	doc_len is the length of the document
'''
def jelinek_smoothing(query, document, doc_len):
	doc_prob = {}
	col_prob = {}
	var_lambda = 0.5
	
	for word in query:
		doc_prob[word] = 0
		for k,v in document.items():
			if word == k:
				doc_prob[word] += v
	
	col_len = 0.0
	
	dir_path = 'D://document//UCL//Data Mining//data//wiki-pages//'
	files_name = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
	
	for word in query:
		col_prob[word] = 0
	
	col_len = 0.0
	for name in files_name:
		document, tmp_len = get_assigned_text(name)
		col_len += tmp_len
		
		for word in query:
			for line in document:
				for k,v in line.items():
					doc_words = re.findall(r'\w+', v)
					cnt = Counter(doc_words)
					if word in doc_words:
						col_prob[word] += cnt[word]
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
		for k,v in document.items():
			if word == k:
				doc_prob[word] += v
	
	col_len = 0.0
	
	dir_path = 'D://document//UCL//Data Mining//data//wiki-pages//'
	files_name = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
	
	for word in query:
		col_prob[word] = 0
	
	col_len = 0.0
	for name in files_name:
		document, tmp_len = get_assigned_text(name)
		col_len += tmp_len
		
		for word in query:
			for line in document:
				for k,v in line.items():
					doc_words = re.findall(r'\w+', v)
					cnt = Counter(doc_words)
					if word in doc_words:
						col_prob[word] += cnt[word]
			print(word+' tf: '+str(col_prob[word]))
			
		print(name+' done, col len: '+str(col_len))
		print('')
		
	result = 0
	for word in query:
		tmp = math.log(((doc_len/(doc_len+const_n)) * (doc_prob[word]/doc_len)) + ((const_n/(doc_len+const_n)) * (col_prob[word]/col_len)) ,10)
		result += tmp
		print(word+': '+str(tmp))
	
	return result