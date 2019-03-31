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


dir_path = 'D://document//UCL//Data Mining//data//wiki-pages'


files_name = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]


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
	claim is the word in that claim, which is a list
	And will return a dict: 'word':'tfidf'
'''
def get_claim_tfidf(claim):
	claim_idf = {}
	claim_tf = {}
	
	claim_cnt = dict(Counter(claim))
	claim = set(claim)
	for word in claim:
		claim_tf[word] = (claim_cnt[word]/len(claim))
	
	dir_path = 'D://document//UCL//Data Mining//data//wiki-pages//'
	files_name = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
	
	for word in claim:
		claim_idf[word] = 0
	
	doc_len = 0
	
	for name in files_name:
		document, tmp_len = get_assigned_text(name)
		doc_len += tmp_len
		print(name+' done! doc len: '+str(doc_len))
		for line in document:
			#print('type: '+str(type(line))+' and: '+str(line))
			for id in line:
				words = re.findall(r'\w+', line[id])
				for word in claim:
					claim_idf[word] += cal_idf(word, words)
				
	for word in claim:
		claim_idf[word] = math.log(doc_len/claim_idf[word] ,10)

	claim_tfidf = {}
	count = 0
	for word in claim:
		claim_tfidf[word] = claim_idf[word] * claim_tf[word]
	
	return claim_tfidf, doc_len

'''
	claim is the word in that claim, and it is a list
	document is a string, doc_len is the length of all documents in the collection
	
'''
def get_doc_tfidf(claim, document, doc_len):
	dir_path = 'D://document//UCL//Data Mining//data//wiki-pages'
	files_name = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
	doc_tf = {}	
	doc_idf = {}

	document = re.findall(r'\w+', document)
	doc_cnt = dict(Counter(document))
	claim = set(claim)	
	intersection = set(claim) & set(document)
	for word in intersection:
		doc_tf[word] = doc_cnt[word]/len(document)
	
	
	for word in intersection:
		doc_idf[word] = 0

	for name in files_name:
		document, tmp_len = get_assigned_text(name)
		print(name+' done!')
		for line in document:
			#print('type: '+str(type(line))+' and: '+str(line))
			for id in line:
				words = re.findall(r'\w+', line[id])
				for word in intersection:
					doc_idf[word] += cal_idf(word, words)

	for word in intersection:
		doc_idf[word] = math.log(doc_len/doc_idf[word], 10)
	
	doc_tfidf = {}
	for word in intersection:
		doc_tfidf[word] = doc_tf[word] * doc_idf[word]
	
	return doc_tfidf
		

def read_file_words(name):
	dir_path = 'D://document//UCL//Data Mining//data//wiki-pages//'
	
	file_words = []
	with io.open(dir_path+name) as file:
		lines = file.readlines()
		for line in lines:
			file_words.append(re.findall(r'\w+',json.loads(line)['text']))
	
	return file_words


'''
	claim and text should be dicts, that claim: 'tfidf', text:'tf,idf' of each word
'''
def get_cosine(claim, text):
	intersection = set(claim.keys()) & set(text.keys())
	numerator = 0
	for word in intersection:
		numerator += (claim[word] * text[word])
		
		print ('word: '+word+' value: '+str(claim[word] * text[word]))
		print('numerator: '+str(numerator))
		print()
	
	var1 = 0
	var2 = 0
	for word in claim:
		var1 += math.pow(claim[word], 2)
	for word in text:
		var2 += math.pow(text[word], 2)
	
	denomiator = math.sqrt(var1) * math.sqrt(var2)
	print('denomiator: '+str(denomiator))
	
	if numerator == 0:
		return 0.0
	else:
		return numerator / denomiator





