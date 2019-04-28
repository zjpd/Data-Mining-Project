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
	
'''
	index is the file name of the assigned text to read
	the method will return a dictionary contains all documetns in a wiki-page and their id
	'id': document
'''
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

	
def get_claim():
	claim_id = [75397, 150448, 214861, 156709, 129629, 33078, 6744, 226034, 40190, 76253]
	train_path = 'D://document//UCL//Data Mining//data'
	claim = []
	
	data = []
	with open(train_path+'//train.jsonl', 'r') as file:
		lines = file.readlines()
		for line in lines:
			tmp = json.loads(line)
			for id in claim_id:
				if tmp['id'] == id:
					data.append(tmp)

	for line in data:
		tmp = line['claim'].lower()
		claim.append(list(set(re.findall(r'\w+', tmp))))

	del data
	gc.collect()
		
	return claim


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
		#print (k+': '+ str(v) +' '+ str(result))
	
	return result

def impl_unigram():
	output_path = 'D://document//UCL//Data Mining//results'
	claim = get_claim()
	for i in range(len(claim)):
		uni_res = []
		doc_id = {}
		for k in range(10):
			uni_res.append(0)
	
	
		dir_path = 'D://document//UCL//Data Mining//data//wiki-pages//'
		files_name = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
		for name in files_name:
			id, doc = get_assigned_text(name, 'text')
			for j in range(len(doc)):
				tmpwords = re.findall(r'\w+', doc[j])
				tmpcnt = Counter(tmpwords)
				if len(tmpwords) == 0:
					continue
				result = unigram_query_likelihood_model(claim[i], tmpcnt, len(tmpwords))
			
				doc_id[id[j]] = result
			print(name+' done')
	
		tmp = {k: v for k, v in sorted(doc_id.items(), key=lambda x: x[1])}
		count = 0
		doc_id = {}
		for k, v in tmp.items():
			doc_id[k] = v
			count += 1
			if count >= 10:
				break
	
		with io.open(output_path+'//unigram'+str(i)+'.txt', 'w', encoding='utf8') as file:
			for k,v in doc_id.items():
				file.write(str(k)+"\t"+str(v)+"\n")
				print(k,v)
		print('claim '+str(i)+' done')

#three choices of model_type: laplace, jelinek and dirichlet
def impl_laplace_jeline_diri(model_type):
	output_path = 'D://document//UCL//Data Mining//results'
	claim = get_claim()
	for i in range(len(claim)):
		uni_res = []
		doc_id = {}
		for k in range(10):
			uni_res.append(0)
	
	
		dir_path = 'D://document//UCL//Data Mining//data//wiki-pages//'
		files_name = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
		for name in files_name:
			id, doc = get_assigned_text(name, 'text')
			for j in range(len(doc)):
				tmpwords = re.findall(r'\w+', doc[j])
				tmpcnt = Counter(tmpwords)
				if len(tmpwords) == 0:
					continue
				if model_type == 'laplace':
					result = laplace_smoothing(claim[i], tmpcnt, len(tmpwords))
				elif: model_type == 'jelinek':
					result = jelinek_smoothing(claim[i], tmpcnt, len(tmpwords))
				elif: model_type == 'dirichlet':
					result = dirichlet_smoothing(claim[i], tmpcnt, len(tmpwords))
					
				doc_id[id[j]] = result
			print(name+' done')
	
		tmp = {k: v for k, v in sorted(doc_id.items(), key=lambda x: x[1])}
		count = 0
		doc_id = {}
		for k, v in tmp.items():
			doc_id[k] = v
			count += 1
			if count >= 10:
				break
	
		with io.open(output_path+'//unigram'+str(i)+'.txt', 'w', encoding='utf8') as file:
			for k,v in doc_id.items():
				file.write(str(k)+"\t"+str(v)+"\n")
				print(k,v)
		print('claim '+str(i)+' done')
	
def laplace_smoothing(query, document, doc_len):
	probability = {}
	uni_word = set()
	
	for word in query:
		probability[word] = 0
		for k,v in document.items():
			uni_word.add(k)
			if word == k:
				probability[word] += v
	
	
	for word in probability:
		probability[word] += 1
	
	result = 1.0
	
	for k,v in probability.items():
		result = result * (v/(doc_len+len(probability)))
	
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