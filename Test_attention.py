import torch
import heapq
import os
import sys
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter
import glob
from pathlib import Path
import random
import pandas as pd
import nltk
import collections
import math
import time
import re
import pickle
from gensim.models.keyedvectors import KeyedVectors
import fnmatch
import codecs
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn.init as weight_init
import gc
import operator
import torch.nn.parameter as Parameter
import torch.optim as optim
import multiprocessing
from sklearn import metrics
from nltk.tokenize import sent_tokenize
from gensim import models
from nltk.corpus import stopwords
from sklearn.metrics import average_precision_score
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from gensim import models
from nltk.corpus import stopwords



# In[2]:


# use_cuda = torch.cuda.is_available()
# torch.backends.cudnn.enabled = True
# gpu = 3
# print(use_cuda)


directory = "/scratchd/home/swayambhu/Time_Stamping/"
# test_folder = "/scratchd/home/swayambhu/Time_Stamping/Test"
# folder = "/scratchd/home/swayambhu/Text_Data/"
container_directory = "/scratchd/home/swayambhu/Time_Stamping/List_of_Documents/"
test_folder = sys.argv[1]


class DocumentContainer(object):
    def __init__(self, sentences, label):
    	self.sentences = sentences
    	self.label = label


# In[5]:


class BatchContainer(object):
    def __init__(self, batch):
        self.batch = batch


# In[6]:


def type_cast_int(input):
    return Variable(torch.from_numpy(np.asarray(input, dtype='int32')).long())


# In[7]:


def type_cast_float(input):
    return Variable(torch.from_numpy(np.asarray(input, dtype='float')).long())


class getEmbeddings(nn.Module):
    def __init__(self, w2v_array, w2vlength, embedding_length):
        super(getEmbeddings, self).__init__()
        self.word_embedding = nn.Embedding(w2vlength, embedding_length, padding_idx = 0)
        self.word_embedding.weight.data.copy_(torch.from_numpy(w2v_array)) ##initializing the embeddings with our own word2vec
        
    def forward(self, x):
        word_embed = self.word_embedding(x)
        return word_embed
## this layer will take the index as input and give the word vectors as output


# In[9]:



class CNNwithPool(nn.Module):
    def __init__(self, cnn_layers, kernel_size0, kernel_size1, kernel_size2):
        super(CNNwithPool, self).__init__()
        self.cnn0 = nn.Conv2d(1, cnn_layers, kernel_size0,stride = 1)
        self.cnn0.bias.data.copy_(weight_init.constant(self.cnn0.bias.data, 0.))
        # self.dropout1 = nn.Dropout(p=0.2)
        self.cnn1 = nn.Conv2d(1, cnn_layers, kernel_size1,stride = 1)
        self.cnn1.bias.data.copy_(weight_init.constant(self.cnn1.bias.data, 0.))
        # self.dropout2 = nn.Dropout(p=0.2)
#        self.cnn2 = nn.Conv2d(1, cnn_layers, kernel_size2,stride = 1)
#        self.cnn2.bias.data.copy_(weight_init.constant(self.cnn2.bias.data, 0.))
#        self.dropout3 = nn.Dropout(p=0.2)
    def forward(self, x):
        cn0 = self.cnn0(x)
        max_pool_size0 = cn0.size(2)
        pooled_list0 = F.max_pool2d(cn0, (max_pool_size0, 1))
        cn1 = self.cnn1(x)
        max_pool_size1 = cn1.size(2)
        pooled_list1 = F.max_pool2d(cn1, (max_pool_size1, 1))
#        cn2 = self.dropout3(self.cnn2(x))
#        print(cn.size())
#        print(cn2.size())
        #        cn_res = torch.cat((cn, cn2), 1)
#        max_pool_size2 = cn2.size(2)
#        pooled_list2 = F.max_pool2d(cn2, (max_pool_size2, 1))
#        temp = torch.cat((pooled_list0, pooled_list1), 1)
#        print(pooled_list1.size())
#        print(pooled_list2.size())
        out = torch.cat((pooled_list0 ,pooled_list1), 1)
        # print(pooled_list0.size() ,pooled_list1.size(), pooled_list2.size(), out.size())
        # print(out.size())
        return out



# In[10]:


class SentenceAttention(nn.Module):
    def __init__(self, embedding_size):
        super(SentenceAttention,self).__init__()
        self.embedding_size = embedding_size
        self.relationMatrix = nn.Linear(self.embedding_size, self.embedding_size, bias = False)
        self.relationVector = nn.Linear(self.embedding_size,1, bias = False)
    def forward(self, x, document_index):
        out = []
        input_dim = x.dim()
        sentence_embedding_size = x.size(1)
        for index, start_end in enumerate(document_index):
#            print(start_end)
            bag = x[start_end[0] : start_end[1]]
            attention_values = self.relationVector(self.relationMatrix(bag))
            keep_attention = F.softmax(attention_values.view((-1,))).view((bag.size(0),1))
            attention_values_softmax = F.softmax(attention_values.view((-1,))).view((bag.size(0),1)).expand(int(start_end[1] - start_end[0]), sentence_embedding_size)
            sentence_vector = attention_values_softmax*bag
            final_sentence_vector = torch.sum(sentence_vector, 0)
            out.append(final_sentence_vector)
        out_concat = torch.cat(out, 0)
        return out_concat, keep_attention


# In[11]:


class ThreeLayerNet(nn.Module):
    def __init__(self, D_in, D_h1, D_h2, D_out):
        super(ThreeLayerNet, self).__init__()
        self.linear1 = nn.Linear(D_in, D_h1, bias = False)
        self.tanh1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.2)
        self.linear2 = nn.Linear(D_h1, D_h2, bias = False)
        self.dropout2 = nn.Dropout(p=0.2)
        self.tanh2 = nn.ReLU()
        self.linear3 = nn.Linear(D_h2, D_out, bias = False)
        # self.linear4 = nn.Linear(30, D_out, bias = False)
    def forward(self, x):
        out_1 = self.dropout1(self.tanh1(self.linear1(x)))
        out_2 = self.dropout2(self.tanh2(self.linear2(out_1)))
        out_3 = self.linear3(out_2)
        # out_4 = self.linear4(out_3)
        return out_3


# In[12]:



class final(nn.Module):
    def __init__(self, w2v_array, w2v_length, embedding_length, cnn_layer, kernel_size0, kernel_size1, kernel_size2, num_of_class, hidden1, hidden2):
        super(final, self).__init__()
        self.w2v_array = w2v_array
        self.w2v_length = w2v_length
        self.embedding_length = embedding_length
        self.kernel_size0 = kernel_size0
        self.kernel_size1 = kernel_size1
        self.kernel_size2 = kernel_size2
        self.cnn_layer = cnn_layer
        self.embedding = getEmbeddings(self.w2v_array, self.w2v_length, self.embedding_length)
        self.cnn = CNNwithPool(self.cnn_layer,self.kernel_size0, self.kernel_size1, self.kernel_size2)
        self.attention = SentenceAttention(2*self.cnn_layer)
        self.num_of_class = num_of_class
        self.D_h1 = hidden1
        self.D_h2 = hidden2
        self.NeuralNet = ThreeLayerNet(2*self.cnn_layer, self.D_h1, self.D_h2, self.num_of_class)
        
    def forward(self, x, document_index):
        embeddings = self.embedding(x).unsqueeze(1)
        cn = self.cnn(embeddings).view((embeddings.size(0), -1))
        attention, given_attention = self.attention(cn, document_index)
        final_output = self.NeuralNet(attention)
        return final_output, given_attention




w2varray = pickle.load(open(directory+'w2varray.p',"rb"))
w2varray = np.asarray(w2varray)
num_of_class = 12
cnn_layer = 200
kernel_size0 = (2,len(w2varray[0]))
kernel_size1 = (3,len(w2varray[0]))
kernel_size2 = (4,len(w2varray[0]))
hidden1 = 512
hidden2 = 128
batch_size = 16
epoch = 5
test_model = final(w2v_array = w2varray, w2v_length = len(w2varray), embedding_length = len(w2varray[0]), cnn_layer = cnn_layer, kernel_size0 = kernel_size0, kernel_size1 = kernel_size1, kernel_size2 = kernel_size2, num_of_class = num_of_class, hidden1 = hidden1, hidden2 = hidden2)

vocabulary = pickle.load(open(directory+'vocabulary.p',"rb"))
# print(test_model)
checkpoint = torch.load(directory+"Models/model_"+str(epoch), map_location=lambda storage, loc: storage)
test_model.load_state_dict(checkpoint['state_dict'])


def get_test_file_path(test_folder, vocabulary, max_sentence_length):
	save_as_list_of_DocumentContainer = []
	for root, dirs, end in os.walk(test_folder):
		for file in end:
			if file.startswith('.'):
				continue
			get_path = os.path.join(root, file)
			save_year = int(file.split('.')[0].split('_')[2][0:4]) - 2000
			save_month = int(file.split('.')[0].split('_')[2][4:6])
			with open(get_path, "r", encoding='utf-8', errors='ignore') as inputFile:
				new_array = []
				content = ""
				for lines in inputFile:
					sent_tokenize_list = sent_tokenize(lines)
					# print(sent_tokenize_list[0])
					for line in sent_tokenize_list:
						content = " ".join(line.split("\t")[0:]) + "\n"
						content = content.lower().strip()
						words = word_tokenize(content)
						filtered_words = [word for word in words if word not in stopwords.words('english')]
						sentence_vector = np.zeros(max_sentence_length)
						index = 0
						for word in filtered_words:
							if word in vocabulary and index < max_sentence_length:
								sentence_vector[index] = vocabulary[word]
								index = index + 1
						sentence_vector = np.asarray(sentence_vector)
						new_array.append(sentence_vector)
					new_array = np.asarray(new_array)
				document_index = []
				doc_begin = 0
				doc_end = doc_begin + len(new_array)
				tup = (doc_begin, doc_end)
				document_index.append(tup)
				input_array = np.asarray(new_array)
				input_array = type_cast_float(input_array)
				result_batch, attention_given = test_model(input_array, document_index)
				result_cpu = result_batch.cpu()
				result_cpu = result_cpu.data.numpy()
				attention_cpu = attention_given.cpu()
				attention_cpu = attention_cpu.data.numpy()
				for i in range(len(attention_given)):
					print(attention_cpu[i], sent_tokenize_list[i])
				print("Score for every month - ", result_cpu)
				print("Actual Class - ", save_year*12 + save_month - 1)
				print("#######################################################")
				# print(result_batch)
				# print(doc_end, new_array)





            # get_path = os.path.join(root, file)
            # save_year = int(file.split('.')[0].split('_')[2][0:4]) - 2000
            # save_month = int(file.split('.')[0].split('_')[2][4:6])
            # with open(get_path, "r", encoding='utf-8', errors='ignore') as inputFile:
            # 	for lines in inputFile:
            #     	sent_tokenize_list = sent_tokenize(lines)
            #     	for line in sent_tokenize_list:
            #         	content = " ".join(line.split("\t")[0:]) + "\n"
            #         	content = content.lower().strip()
            #         	words = word_tokenize(content)

list_of_train_docs = pickle.load(open(container_directory + "list_of_documents_0.p", "rb"))
max_sentence_length = len(list_of_train_docs[0].sentences[0])
test_file_path = get_test_file_path(test_folder, vocabulary, max_sentence_length)
