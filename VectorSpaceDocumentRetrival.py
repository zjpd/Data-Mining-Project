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

claim_words = []
for line in claim:
	words = re.findall(r'\w+', line)
	claim_words.append(words)

claim = dict(Counter(claim_words[0]))
	
file_text = []
for i in range(len(files_name)):
	with open(dir_path+'//'+files_name[i]) as file:
		lines = file.readlines()
		for line in lines:
			file_text.append(json.loads(line)['text'])


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
'''
	claim is the word in that claim, and words are all the words in all the claims, claim is dict
	And will return a dict: 'words':'tfidf'
	claim_len is the number of the words that the claim has
'''
def get_claim_tfidf(claim, words):
	claim_idf = []
	for k,v in claim.items():
		idf_count = 0
		for j in range(len(words)):
			if k in words[j]:
				idf_count += 1
				break
		claim_idf.append(math.log(len(words)/idf_count, 10))

	claim_tf = []
	for k,v in claim.items():
		claim_tf.append(v/len(words))
	

	claim_tfidf = {}
	count = 0
	for word in claim:
		claim_tfidf[word] = claim_idf[count] * claim_tf[count]
		count += 1
	
	return claim_tfidf

'''
	claim is the word in that claim, and it is a dictionary
	file_text is the word in the assigned text, and it is a dictionary
	
'''
def get_text_tfidf(claim, file_text, text_len):
	dir_path = 'D://document//UCL//Data Mining//data//wiki-pages'
	files_name = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
	
	intersection = set(claim.keys()) & set(file_text.keys())
	text_tf = []
	for word in intersection:
		text_tf.append(file_text[word]/text_len)
	
	text_idf = {}
	for word in intersection:
		text_idf[word] = 0

	for name in files_name:
		file_words = read_file_words(name)
		for word in intersection:
			for i in range(len(file_words)):
				if word in file_words[i]:
					text_idf[word] += 1
					break
		print(name+' done!')

	for word in text_idf:
		text_idf[word] = math.log(len(files_name)/text_idf[word], 10)
	
	text_tfidf = {}
	count = 0
	for word in text_idf:
		text_tfidf[word] = text_tf[count] * text_idf[word]
		count += 1
	
	return text_tfidf
		

def read_file_words(name):
	dir_path = 'D://document//UCL//Data Mining//data//wiki-pages//'
	
	file_words = []
	with io.open(dir_path+name) as file:
		lines = file.readlines()
		for line in lines:
			file_words.append(re.findall(r'\w+',json.loads(line)['text']))
	
	return file_words
'''
	get the words from the assigned text
'''
def get_assigned_text(index):
	dir_path = 'D://document//UCL//Data Mining//data//wiki-pages'
	file_text = []
	with io.open(dir_path+'//wiki-'+index+'.jsonl') as file:
		lines = file.readlines()
		for line in lines:
			file_text.append(json.loads(line)['text'])

	tmpstr = ''
	for line in file_text:
		tmpstr += line

	words = re.findall(r'\w+', tmpstr)
	file_text = dict(Counter(words))
	text_len = len(words)
	
	del tmpstr
	del words
	gc.collect()
	
	return file_text, text_len
	



