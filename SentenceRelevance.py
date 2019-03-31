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

document, doc_len = get_assigned_text('wiki-001.jsonl')
count = 0
sentences = []
for line in document:
	if count >= 5:
		break
	for k,v in line.items():
		sentences.append(re.findall(r'\w+', v))
	count += 1

#path = get_tmpfile("word2vec.model")
model = Word2Vec(sentences, size=100, window=5, min_count=1, workers=4)
model.save("word2vec.model")

vector = []
intersection = set(claim[0]) & set(sentences)
for word in intersection:
	vector.append(model[word])

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