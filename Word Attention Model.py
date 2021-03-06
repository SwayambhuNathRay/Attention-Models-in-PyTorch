
# coding: utf-8

# In[4]:


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


# In[5]:


use_cuda = torch.cuda.is_available()
torch.backends.cudnn.enabled = True
gpu = 2
print(use_cuda)


# In[6]:


directory = "/scratchd/home/swayambhu/Time_Stamping/Dumped_files/Word_Attn_40k/"
container_directory = "/scratchd/home/swayambhu/Time_Stamping/Dumped_files/Word_Attn_40k/List_of_Documents/"
test_directory = "/scratchd/home/swayambhu/Time_Stamping/Dumped_files/Word_Attn_40k/Test_Documents/"


# In[7]:


class DocumentContainer(object):
    def __init__(self, sentences, label):
        self.sentences = sentences
        self.label = label

class BatchContainer(object):
    def __init__(self, batch):
        self.batch = batch

def type_cast_int(input):
    return Variable(torch.from_numpy(np.asarray(input, dtype='int32')).long().cuda(gpu))

def type_cast_float(input):
    return Variable(torch.from_numpy(np.asarray(input, dtype='float')).long().cuda(gpu))


# In[8]:


class getEmbeddings(nn.Module):
    def __init__(self, w2v_array, w2vlength, embedding_length):
        super(getEmbeddings, self).__init__()
        self.word_embedding = nn.Embedding(w2vlength, embedding_length, padding_idx = 0)
        self.word_embedding.weight.data.copy_(torch.from_numpy(w2v_array))

    def forward(self, x):
        word_embed = self.word_embedding(x)
        return word_embed


# In[20]:


class WordAttention(nn.Module):
    def __init__(self, word_embedding_size):
        super(WordAttention,self).__init__()
        self.embedding_size = word_embedding_size
        self.relationMatrix = nn.Linear(self.embedding_size, self.embedding_size, bias = False)
        self.relationVector = nn.Linear(self.embedding_size,1, bias = False)
    def forward(self, x, word_index):
        out = []
        input_dim = x.dim()
        word_embedding_size = x.size(2)
        for index in range(len(word_index)):
            tup = word_index[index]
            bag = x[index][tup[0] : tup[1]]
            attention_values = self.relationVector(self.relationMatrix(bag))
            keep_attention = F.softmax(attention_values.view((-1,))).view((bag.size(0),1))
            attention_values_softmax = F.softmax(attention_values.view((-1,))).view((bag.size(0),1)).expand(int(tup[1] - tup[0]), word_embedding_size)
            sentence_vector = attention_values_softmax*bag
            final_sentence_vector = torch.sum(sentence_vector, 0)
            out.append(final_sentence_vector)
        out_concat = torch.stack(out)
        return out_concat, keep_attention


# In[27]:


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
            bag = x[start_end[0] : start_end[1]]
            attention_values = self.relationVector(self.relationMatrix(bag))
            keep_attention = F.softmax(attention_values.view((-1,))).view((bag.size(0),1))
            attention_values_softmax = F.softmax(attention_values.view((-1,))).view((bag.size(0),1)).expand(int(start_end[1] - start_end[0]), sentence_embedding_size)
            sentence_vector = attention_values_softmax*bag
            final_sentence_vector = torch.sum(sentence_vector, 0)
            out.append(final_sentence_vector)
        out_concat = torch.stack(out)
        return out_concat, keep_attention


# In[28]:


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
    def forward(self, x):
        out_1 = self.dropout1(self.tanh1(self.linear1(x)))
        out_2 = self.dropout2(self.tanh2(self.linear2(out_1)))
        out_3 = self.linear3(out_2)
        return out_3


# In[29]:


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
        self.word_attention = WordAttention(self.embedding_length)
        self.attention = SentenceAttention(embedding_length)
        self.num_of_class = num_of_class
        self.D_h1 = hidden1
        self.D_h2 = hidden2
        self.NeuralNet = ThreeLayerNet(embedding_length, self.D_h1, self.D_h2, self.num_of_class)
    def forward(self, x, document_index, word_index):
        embeddings = self.embedding(x)#.unsqueeze(1)
        cn, word_attentions_given = self.word_attention(embeddings, word_index)#.view((embeddings.size(0), -1))
        attention, given_attention = self.attention(cn, document_index)
        final_output = self.NeuralNet(attention)
        return final_output, given_attention


# In[30]:


def accuracy(result_batch, class_label, num_of_class):
    batch_size = result_batch.size(0)
    result_cpu = result_batch.cpu()
    result_cpu = result_cpu.data.numpy()
    compare_performance = []
    class_label_cpu = class_label.cpu()
    sum_ = 0
    length = len(class_label_cpu)
    right = 0
    class_label_cpu = class_label_cpu.data.numpy()
    for i in range(batch_size):
        keep_actual = class_label_cpu[i]
        keep_score = result_cpu[i]
        keep_class = 0
        max_element = -1000000
        for j in range(len(keep_score)):
            if(keep_score[j] > max_element):
                keep_class = j
                max_element = keep_score[j]
        sum_ = sum_ + abs(keep_actual - keep_class)
        if int(keep_actual - keep_class) is 0:
            right = right + 1
    return sum_, right, length, keep_class


# # Training and Testing in a single loop

# In[31]:


def trainModel_mod_mod(container_directory, w2v_array, w2v_length, embedding_length, cnn_layer, kernel_size0, kernel_size1, kernel_size2, batch_size, num_of_class, hidden_layer_1, hidden_layer_2, epochs):
    model = final(w2v_array = w2varray, w2v_length = len(w2varray), embedding_length = len(w2varray[0]), cnn_layer = cnn_layer, kernel_size0 = kernel_size0, kernel_size1 = kernel_size1, kernel_size2 = kernel_size2, num_of_class = int(num_of_class), hidden1 = hidden_layer_1, hidden2 = hidden_layer_2).cuda(gpu)
    print(model)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = 0.001)
    prev_learning_rate = 0.1
    loss_function = nn.CrossEntropyLoss().cuda(gpu)
    min_avg_mean = 12
    max_avg_acc = 0
    min_avg_mean_epoch = -1
    max_avg_acc_epoch = -1
    num_of_test_files = 0
    sentence_length = 2000
    keep_container_files = []
    for root, dirs, end in os.walk(container_directory):
        for file in end:
            get_path = os.path.join(root, file)
            keep_container_files.append(get_path)
    num_of_container = len(keep_container_files)
    for epoch in range(epochs):
        no_of_rights = 0
        deviation = 0
        no_of_files = 0
        train_length = 0
        random.shuffle(keep_container_files)
        total_loss = 0
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        print(str(now))
        for i in range(num_of_container):
            container = pickle.load(open(keep_container_files[i],"rb"))
            number_of_docs = 0
            for index in container:
                number_of_docs = number_of_docs+1
            totalBatches = int(math.ceil(number_of_docs/batch_size))
            num = [z%number_of_docs for z in range(totalBatches*batch_size)]
            random.shuffle(num)
            containBatch = []
            for i in range(totalBatches):
                temp_batch = num[i*batch_size:i*batch_size+batch_size]
                temp_batch = np.asarray(temp_batch)
                store = BatchContainer(temp_batch)
                containBatch.append(store)
            for batches in containBatch:
                input_array = []
                class_label = []
                document_index = []
                word_index = []
                doc_in_batch = batches.batch
                doc_start = 0
                word_start = 0
                word_end = 0
                num_of_items = 0
                for elems in doc_in_batch:
                    keep_doc = container[elems]
                    if int(keep_doc.sentences.shape[0]) is 0:
                        continue
                    for sente in range(len(keep_doc.sentences)):
                        word_start = 0 * len(keep_doc.sentences[0])
                        word_end = 0
                        for inds in range(len(keep_doc.sentences[0])):
                            if int(keep_doc.sentences[sente][inds]) is not 0:
                                word_end = word_end + 1;
                        if word_end is 0:
                            word_end = 1
                        word_end = word_start + word_end
                        tup = (word_start, word_end)
                        word_index.append(tup)
                        num_of_items = num_of_items + 1
                    doc_end = keep_doc.sentences.shape[0] + doc_start
                    tup = (doc_start, doc_end)
                    document_index.append(tup)
                    doc_start = doc_end
                    class_label.append(keep_doc.label - 1)
                    for rows in keep_doc.sentences:
                        input_array.append(rows)
                        max_length_sentence = len(rows)
                no_of_files = no_of_files + len(input_array)
                input_array = np.asarray(input_array)
                class_label = np.asarray(class_label).reshape(-1,1)
                input_array = type_cast_float(input_array)
                class_label = type_cast_float(class_label)
                result_batch, attention_given = model(input_array, document_index, word_index)
                class_label = class_label.squeeze(1)
                loss = loss_function(result_batch, class_label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss+=loss.data
                dev, sup, leng, clss = accuracy(result_batch, class_label, num_of_class)
                no_of_rights = no_of_rights + sup
                deviation = deviation + dev
                train_length = train_length + leng
                del input_array
                del class_label
                del document_index
            del container
            del loss
            gc.collect()
        print("########", epoch, "#######")
        print(total_loss)
        print("Training accuracy - ", no_of_rights/train_length)
        print("Training deviation - ", deviation/train_length)
        torch.save({'epoch': epoch,'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, directory+"Models/model_"+str(epoch))
        container_test = pickle.load(open(test_directory+"list_of_full_test_documents"+str(sentence_length)+".p","rb"))
        num_of_test_files = len(container_test)
        files_path = pickle.load(open(test_directory + "all_test_file_path" + str(sentence_length)+".p", "rb"))
        input_array = []
        class_label = []
        document_index = []
        doc_start = 0
        count = 0
        flag = 0
        sum_ = 0
        num_of_rights = 0
        keep_num_of_rights = 0
        for elems in range(int(num_of_test_files)):
            count = count + 1
            keep_doc = container_test[elems]
            del input_array
            del class_label
            del document_index
            input_array = []
            class_label = []
            word_index = []
            document_index = []
            word_start = 0
            word_end = 0
            if int(keep_doc.sentences.shape[0]) is 0:
                continue
            for sente in range(len(keep_doc.sentences)):
                word_start = 0 * len(keep_doc.sentences[0])
                word_end = 0
                for inds in range(len(keep_doc.sentences[0])):
                    if int(keep_doc.sentences[sente][inds]) is not 0:
                        word_end = word_end + 1;
                if word_end is 0:
                    word_end = 1
                word_end = word_start + word_end
                tup = (word_start, word_end)
                word_index.append(tup)
            del input_array
            del class_label
            input_array = []
            class_label = []
            del document_index
            document_index = []
            doc_start = 0
            doc_end = keep_doc.sentences.shape[0] + doc_start
            class_label.append(keep_doc.label - 1)
            for rows in keep_doc.sentences:
                input_array.append(rows)
            tup = (doc_start, doc_end)
            document_index.append(tup)
            input_array = np.asarray(input_array)
            class_label = np.asarray(class_label).reshape(-1,1)
            input_array = type_cast_float(input_array)
            class_label = type_cast_float(class_label)
            test_result, attention_given = model(input_array, document_index, word_index)
            mean_abs_deviation, num_of_rights, lgth, clss = accuracy(test_result, class_label, num_of_class)
            sum_ = sum_+ mean_abs_deviation
            keep_num_of_rights = keep_num_of_rights + num_of_rights
        print("epoch - ", epoch)
        print("Avg mean on validation - ", sum_/count)
        print("Avg accuracy on validation - ", keep_num_of_rights/count * 100)
        if min_avg_mean > sum_/count:
            min_avg_mean = sum_/count
            min_avg_mean_epoch = epoch
        if max_avg_acc < keep_num_of_rights/count * 100:
            max_avg_acc = keep_num_of_rights/count * 100
            max_avg_acc_epoch = epoch
    print("Min deviation obtained ", min_avg_mean, " at epoch ", min_avg_mean_epoch)
    print("Max accuracy obtained ", max_avg_acc, " at epoch ", max_avg_acc_epoch)
    print("number of files - ", num_of_test_files)
    return model, max_length_sentence


# In[32]:


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
partition = 2000
sentence_length = 2000
model = final(w2v_array = w2varray, w2v_length = len(w2varray), embedding_length = len(w2varray[0]), cnn_layer = cnn_layer,kernel_size0 = kernel_size0, kernel_size1 = kernel_size1, kernel_size2 = kernel_size2, num_of_class = num_of_class, hidden1 = hidden1, hidden2 = hidden2).cuda(gpu)
epochs = 20
model, max_length_sentence = trainModel_mod_mod(container_directory = container_directory, w2v_array = w2varray, w2v_length = len(w2varray), embedding_length = len(w2varray[0]), cnn_layer = cnn_layer, kernel_size0 = kernel_size0, kernel_size1 = kernel_size1, kernel_size2 = kernel_size2, batch_size = batch_size, num_of_class = num_of_class, hidden_layer_1 = hidden1, hidden_layer_2 = hidden2, epochs = epochs)


# In[ ]:




