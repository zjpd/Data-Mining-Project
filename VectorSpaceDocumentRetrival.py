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

'''
	return all the required claims in a list
'''	
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
		claim.append(re.findall(r'\w+', tmp))

	del data
	gc.collect()
		
	return claim

'''
	word is a word to calculate its idf
	words are the document to justify whether it contains the word
	if contains return 1
	else return 0
'''
def cal_idf(word, words):
	if word in words:
		return 1
	else:
		return 0


'''
	claim is a list contains 10 claims, each claim is a list contains words in that claim
	And will return a list contains 10 dict: 'word':'tfidf'
'''
def get_claim_tfidf(claim):
	claim_idf = []
	claim_tf = []
	
	for i in range(len(claim)):
		claim_cnt = dict(Counter(claim[i]))
		tmp = set(claim[i])
		tmpdict = {}
		for word in tmp:
			tmpdict[word] = claim_cnt[word] / len(tmp)
		claim_tf.append(tmpdict)
	
	
	dir_path = 'D://document//UCL//Data Mining//data//wiki-pages//'
	files_name = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
	
	for i in range(len(claim)):
		tmpdict = {}
		for word in claim[i]:
			tmpdict[word] = 0
		claim_idf.append(tmpdict)

	doc_len = 0
	for name in files_name:
		id, doc = get_assigned_text(name, 'text')
		doc_len += len(doc)
		print(name+' done! doc len: '+str(doc_len))
		for i in range(len(doc)):
			words = re.findall(r'\w+', doc[i])
			for j in range(len(claim)):
				for word in claim[j]:
					claim_idf[j][word] += cal_idf(word, words)
				
	for i in range(len(claim)):
		tmp = set(claim[i])
		for word in tmp:
			try:
				claim_idf[i][word] = math.log(doc_len/claim_idf[i][word] ,10)
			except ZeroDivisionError:
				print(word, claim_idf[i][word])
			#break
		
	claim_tfidf = []
	count = 0
	for i in range(len(claim)):
		tmp = list(set(claim[i]))
		tmpdict = {}
		for word in tmp:
			tmpdict[word] = claim_idf[i][word] * claim_tf[i][word]
		claim_tfidf.append(tmpdict)
		
	return claim_tfidf, claim_idf
		

'''
	claim and text should be dicts, that claim: 'tfidf', text:'tfidf' of each word
'''
def get_cosine(claim, text):
	intersection = set(claim.keys()) & set(text.keys())
	numerator = 0
	for word in intersection:
		numerator += (claim[word] * text[word])
		
	var1 = 0
	var2 = 0
	for word in claim.keys():
		var1 += math.pow(claim[word], 2)
	for word in text.keys():
		var2 += math.pow(text[word], 2)
	
	denomiator = math.sqrt(var1) * math.sqrt(var2)
	
	if numerator == 0:
		return 0.0
	else:
		return numerator / denomiator

		
def cal_tfidf():
	dir_path = 'D://document//UCL//Data Mining//data//wiki-pages'
	output_path = 'D://document//UCL//Data Mining//results'
	files_name = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
	claim = get_claim()
	claim_tfidf, claim_idf = get_claim_tfidf(claim)
	
	for i in range(len(claim)):
		cos_res = []
		doc_id = {}
		for k in range(10):
			cos_res.append(0)
		
		for name in files_name:
			id, doc = get_assigned_text(name, 'text')
			for j in range(len(doc)):
				words = re.findall(r'\w+', doc[j])
				doc_cnt = dict(Counter(words))
				intersection = set(words) & set(claim[i])
				tmp_tf = {}
				tmp_idf = {}
				tmp_tfidf = {}
				for word in intersection:
					tmp_tf[word] = doc_cnt[word]/len(doc[j])
					tmp_idf[word] = claim_idf[i][word]
					tmp_tfidf[word] = tmp_tf[word] * tmp_idf[word]
		
				cosine = get_cosine(claim_tfidf[i], tmp_tfidf)
		
				cos_res.sort()
				if cosine > cos_res[0]:
					for k,v in doc_id.copy().items():
						if cos_res[0] == v:
							del doc_id[k]
					cos_res = cos_res[1:]
					cos_res.append(cosine)
					doc_id[id[j]] = cosine
			print(name+' done')
		
		with io.open(output_path+'//cosine'+str(i)+'.txt', 'w', encoding='utf8') as file:
			for k,v in doc_id.items():
				file.write(str(k)+"\t"+str(v)+"\n")
				print(k,v)
		print('claim '+str(i)+' done')






		
		
