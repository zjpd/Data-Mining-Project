from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
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
from numpy import array, newaxis


'''
	1. get train data that includes claim in train.jsonl
	2. get the evidences of the claim and retrieve the relevant sentences
	3. label the sentences
	4. setup the model
	
'''

train_path = 'D://document//UCL//Data Mining//data'
data = []
with open(train_path+'//train.jsonl', 'r') as file:
	lines = file.readlines()
	for line in lines:
		tmp = json.loads(line)
		if tmp['label'] != 'NOT ENOUGH INFO' :
			data.append(tmp)

			
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
	
	tmp[line['id']] = tmplist
	evidences.append(tmp)


x_train = []
y_train = []



def get_train_data():


