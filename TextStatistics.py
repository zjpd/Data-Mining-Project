import json
import random
import re
import os
import sys
import re
import io
import gc
from tqdm import tqdm
from collections import Counter
from nltk import word_tokenize
from os import listdir
from os.path import isfile, join

wiki_path = 'D://document//UCL//Data Mining//data//wiki-pages//wiki-001.jsonl'
dir_path = 'D://document//UCL//Data Mining//data//wiki-pages'
output_path = 'D://document//UCL//Data Mining//results'

files_name = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]

data = []
for i in range(len(files_name)):
	with open(dir_path+'//'+files_name[i], 'r') as file:
		lines = file.readlines()
		for line in lines:
			data.append(json.loads(line))


text = []		
for i in range(len(data)):
	text.append((data[i]['text']).lower())

del data
gc.collect()

words = re.findall(r'\w+', text[0])
last_cnt = Counter(words)
cnt = Counter(words)
fre_result = Counter(words)
count = 0
for i in range(len(text)):
	if i+1 >= len(text):
		break
	words = re.findall(r'\w+', text[i+1])
	cnt = Counter(words)
	fre_result = cnt+fre_result
	#print(str(len(cnt))+"   shit")
	#print(len(fre_result))
	#print(cnt.most_common(5))
	print(count)
	last_cnt = cnt
	
	if count%10000 == 0:
		write_txt(fre_result, count)
		fre_result = Counter()
	
	count += 1
		
def write_txt(cnt, count):
	f_write = io.open(output_path+'//frequences'+str(count)+'.txt','w',encoding='utf8')
	for k,v in cnt.items():
		f_write.write(str(k)+"\t"+str(v)+"\r\n")
		f_write.flush()
	f_write.close()

fwrite = open("D://document//UCL//Data Mining//data//test.txt",'w')
for k,v in cnt.items():
	fwrite.write(str(k)+"\t"+str(v)+"\r\n")
	fwrite.flush()
fwrite.close()


def read_all_wiki(dir_path):
	files_name = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
	return files_name

def term_frequency():
	dir_path = 'D://document//UCL//Data Mining//data//wiki-pages'
	files_name = read_all_wiki(dir_path)
	text = []
	data = []
	
	for i in range(len(files_name)):
		with open(dir_path+'//'+files_name[i], 'r') as file:
			lines = file.readlines()
			for line in lines:
				data.append(json.loads(line))
	
	for i in range(len(data)):
		text.append((data[i]['text']).lower())
	
	for i in range(len(text)):
		words = re.findall(r'\w+', text[i])
		cnt = Counter(words)
	
	f_write = io.open(dir_path+'//frequences.txt','w',encoding='utf8')
	for k,v in cnt.items():
		f_write.write(str(k)+"\t"+str(v)+"\r\n")
		f_write.flush()
	f_write.close()