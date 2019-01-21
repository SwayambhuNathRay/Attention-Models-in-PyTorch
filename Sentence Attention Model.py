
# coding: utf-8

# In[13]:


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


# In[14]:


use_cuda = torch.cuda.is_available()
torch.backends.cudnn.enabled = True
gpu = 2
print(use_cuda)


# In[15]:


directory = "/scratchd/home/swayambhu/Time_Stamping/Dumped_files/Sent_Attn_CNN_25k/"
container_directory = "/scratchd/home/swayambhu/Time_Stamping/Dumped_files/Sent_Attn_CNN_25k/List_of_Documents/"
test_directory = "/scratchd/home/swayambhu/Time_Stamping/Dumped_files/Sent_Attn_CNN_25k/Test_Documents/"


# In[16]:


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


# In[17]:


class getEmbeddings(nn.Module):
    def __init__(self, w2v_array, w2vlength, embedding_length):
        super(getEmbeddings, self).__init__()
        self.word_embedding = nn.Embedding(w2vlength, embedding_length, padding_idx = 0)
        self.word_embedding.weight.data.copy_(torch.from_numpy(w2v_array)) ##initializing the embeddings with our own word2vec
        
    def forward(self, x):
        word_embed = self.word_embedding(x)
        return word_embed


# In[18]:


class CNNwithPool(nn.Module):
    def __init__(self, cnn_layers, kernel_size0, kernel_size1, kernel_size2):
        super(CNNwithPool, self).__init__()
        self.cnn0 = nn.Conv2d(1, cnn_layers, kernel_size0,stride = 1)
        self.cnn0.bias.data.copy_(weight_init.constant(self.cnn0.bias.data, 0.))
        self.cnn1 = nn.Conv2d(1, cnn_layers, kernel_size1,stride = 1)
        self.cnn1.bias.data.copy_(weight_init.constant(self.cnn1.bias.data, 0.))
#        self.cnn2 = nn.Conv2d(1, cnn_layers, kernel_size2,stride = 1)
#        self.cnn2.bias.data.copy_(weight_init.constant(self.cnn2.bias.data, 0.))
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


# In[19]:


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
            attention_values = self.relationVector(F.tanh(self.relationMatrix(bag)))
            keep_attention = F.softmax(attention_values.view((-1,))).view((bag.size(0),1))
            attention_values_softmax = F.softmax(attention_values.view((-1,))).view((bag.size(0),1)).expand(int(start_end[1] - start_end[0]), sentence_embedding_size)
            sentence_vector = attention_values_softmax*bag
            final_sentence_vector = torch.sum(sentence_vector, 0)
            out.append(final_sentence_vector)
        out_concat =  torch.stack(out)
        return out_concat, keep_attention


# In[20]:


class ThreeLayerNet(nn.Module):
    def __init__(self, D_in, D_h1, D_h2, D_out):
        super(ThreeLayerNet, self).__init__()
        self.linear1 = nn.Linear(D_in, D_h1, bias = False)
        self.dropout1 = nn.Dropout(p=0.2)
        self.linear2 = nn.Linear(D_h1, D_h2, bias = False)
        self.dropout2 = nn.Dropout(p=0.2)
        self.linear3 = nn.Linear(D_h2, D_out, bias = False)
    def forward(self, x):
        out_1 = self.dropout1(F.relu(self.linear1(x)))
        out_2 = self.dropout2(F.relu(self.linear2(out_1)))
        out_3 = self.linear3(out_2)
        return out_3


# In[21]:


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


# In[22]:


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


# In[23]:


def trainModel_mod_mod(container_directory, w2v_array, w2v_length, embedding_length, cnn_layer, kernel_size0, kernel_size1, kernel_size2, batch_size, num_of_class, hidden_layer_1, hidden_layer_2, epochs):
    model = final(w2v_array = w2varray, w2v_length = len(w2varray), embedding_length = len(w2varray[0]), cnn_layer = cnn_layer, kernel_size0 = kernel_size0, kernel_size1 = kernel_size1, kernel_size2 = kernel_size2, num_of_class = int(num_of_class), hidden1 = hidden_layer_1, hidden2 = hidden_layer_2).cuda(gpu)
    print(model)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = 0.001)
    prev_learning_rate = 0.1
    loss_function = nn.CrossEntropyLoss().cuda(gpu)
    keep_container_files = []
    for root, dirs, end in os.walk(container_directory):
        for file in end:
            get_path = os.path.join(root, file)
            keep_container_files.append(get_path)
    num_of_container = len(keep_container_files)
    for epoch in range(epochs):
        # if epoch >= 9:
        #     for p in model.embedding.parameters():
        #         p.requires_grad = False
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
                doc_in_batch = batches.batch
                doc_start = 0
                for elems in doc_in_batch:
                    keep_doc = container[elems]
                    if int(keep_doc.sentences.shape[0]) is 0:
                        continue
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
                result_batch, attention_given = model(input_array, document_index)
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
    return model, max_length_sentence


# In[24]:


w2varray = pickle.load(open(directory+'w2varray.p',"rb"))
vocabulary = pickle.load(open(directory+'vocabulary.p',"rb"))
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
epochs = 100
model, max_length_sentence = trainModel_mod_mod(container_directory = container_directory, w2v_array = w2varray, w2v_length = len(w2varray), embedding_length = len(w2varray[0]), cnn_layer = cnn_layer, kernel_size0 = kernel_size0, kernel_size1 = kernel_size1, kernel_size2 = kernel_size2, batch_size = batch_size, num_of_class = num_of_class, hidden_layer_1 = hidden1, hidden_layer_2 = hidden2, epochs = epochs)


# # Testing the model without any partitioning 

# In[ ]:


test_model = final(w2v_array = w2varray, w2v_length = len(w2varray), embedding_length = len(w2varray[0]), cnn_layer = cnn_layer, kernel_size0 = kernel_size0, kernel_size1 = kernel_size1, kernel_size2 = kernel_size2, num_of_class = num_of_class, hidden1 = hidden1, hidden2 = hidden2).cuda(gpu)
print(test_model)

min_avg_mean = 12
max_avg_acc = 0
min_avg_mean_epoch = -1
max_avg_acc_epoch = -1
num_of_test_files = 0

for epoch in range(epochs):
# for ep in range(0):
   # epoch = 99
    checkpoint = torch.load(directory+"Models/model_"+str(epoch), map_location=lambda storage, loc: storage)
    test_model.load_state_dict(checkpoint['state_dict'])
    test_model.eval()
    container = pickle.load(open(test_directory+"list_of_full_test_documents"+str(sentence_length)+".p","rb"))
    files_path = pickle.load(open(test_directory + "development_file_path" + str(sentence_length)+".p", "rb"))
    num_of_test_files = len(container)
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
        keep_doc = container[elems]
        del input_array
        del class_label
        del document_index
        input_array = []
        class_label = []
        document_index = []
        if int(keep_doc.sentences.shape[0]) is 0:
            continue
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
        cass_label = type_cast_float(class_label)
        test_result, attention_given = test_model(input_array, document_index)
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
    print("#######################################################################")
print("Min deviation obtained ", min_avg_mean, " at epoch ", min_avg_mean_epoch)
print("Max accuracy obtained ", max_avg_acc, " at epoch ", max_avg_acc_epoch)
print("number of files - ", num_of_test_files)


# # Testing the Model with partition 

# In[ ]:


partition = 5


min_avg_mean = 12
max_avg_acc = 0
min_avg_mean_epoch = -1
max_avg_acc_epoch = -1
num_of_test_files = 0

for epoch in range(epochs):
# for ep in range(0):
    # epoch = 99
    checkpoint = torch.load(directory+"Models/model_"+str(epoch), map_location=lambda storage, loc: storage)
    test_model.load_state_dict(checkpoint['state_dict'])
    test_model.eval()
    container = pickle.load(open(test_directory+"list_of_full_test_documents"+str(sentence_length)+".p","rb"))
    files_path = pickle.load(open(test_directory + "all_test_file_path" + str(sentence_length)+".p", "rb"))
    num_of_test_files = len(container)
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
        keep_doc = container[elems]
        del input_array
        del class_label
        del document_index
        input_array = []
        class_label = []
        document_index = []
        if int(keep_doc.sentences.shape[0]) is 0:
            continue
        if keep_doc.sentences.shape[0] < partition:
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
            # print(input_array.shape)
            class_label = np.asarray(class_label).reshape(-1,1)
            input_array = type_cast_float(input_array)
            class_label = type_cast_float(class_label)
            test_result, attention_given = test_model(input_array, document_index)
            # print(attention_given, keep_file_path[elems][0])
            # pickle.dump(test_result, open(directory+"results" + str(int(elems/40)) +".p", "wb"))
            # pickle.dump(class_label, open(directory+"actual" + str(int(elems/40)) +".p", "wb"))
            mean_abs_deviation, num_of_rights, lgth, clss = accuracy(test_result, class_label, num_of_class)
            sum_ = sum_+ mean_abs_deviation
            keep_num_of_rights = keep_num_of_rights + num_of_rights
        else:
            keep_most_attentive_sentences = {}
            number_of_sentences = keep_doc.sentences.shape[0]
            get_sentences = keep_doc.sentences
            number_of_partition = number_of_sentences - partition + 1
            keep_output = np.zeros(num_of_class)
            actual_class = keep_doc.label - 1
            for i in range(number_of_partition):
                del input_array
                del class_label
                input_array = []
                class_label = []
                del document_index
                document_index = []
                doc_start = 0
                input_array = get_sentences[i:i+partition]
                doc_end = doc_start + partition
                class_label.append(keep_doc.label - 1)
                tup = (doc_start, doc_end)
                document_index.append(tup)
                input_array = np.asarray(input_array)
                class_label = np.asarray(class_label).reshape(-1,1)
                input_array = type_cast_float(input_array)
                class_label = type_cast_float(class_label)
                test_result, attention_given = test_model(input_array, document_index)
                max_attentive_sentence = -1
                attention_till_now = -1
                max_attention_given = attention_given[0]
                attention_given = attention_given.data.cpu().numpy()
                # print(attention_given)
                j = 0
                for j in range(len(attention_given)):
                    if attention_given[j] >= max_attention_given:
                        max_attention_given = attention_given[j]
                        max_attentive_sentence = j
                        # print(j)


                if (i+max_attentive_sentence) not in keep_most_attentive_sentences:
                    keep_most_attentive_sentences[i+max_attentive_sentence] = get_sentences[i+max_attentive_sentence]
                devi, rights, lgth, output_class = accuracy(test_result, class_label, num_of_class)
#             while len(keep_most_attentive_sentences) > partition:
#                 sentence_array = []
#                 for g in keep_most_attentive_sentences:
#                     sentence_array.append(keep_most_attentive_sentences[g])
#                 del keep_most_attentive_sentences
#                 keep_most_attentive_sentences = {}
#                 number_of_sentences = len(sentence_array)
#                 number_of_partition = number_of_sentences - partition + 1
#                 input_array = []
#                 document_index = []
#                 doc_start = 0
#                 doc_end = 0
#                 for i in range(number_of_partition):
#                     input_array = sentence_array[i:i+partition]
#                     doc_end = doc_start + len(input_array)
#                     tup = (doc_start, doc_end)
#                     document_index.append(tup)
#                     input_array = np.asarray(input_array)
#                     input_array = type_cast_float(input_array)
#                     test_result, attention_given = test_model(input_array, document_index)
#                     max_attentive_sentence = -1
#                     attention_till_now = -1
#                     max_attention_given = attention_given[0]
#                     attention_given = attention_given.data.cpu().numpy()
#                 # print(attention_given)
#                     j = 0
#                     for j in range(len(attention_given)):
#                         if attention_given[j] >= max_attention_given:
#                             max_attention_given = attention_given[j]
#                             max_attentive_sentence = j
#                     if (i+max_attentive_sentence) not in keep_most_attentive_sentences:
#                         keep_most_attentive_sentences[i+max_attentive_sentence] = sentence_array[i+max_attentive_sentence]
# #                keep_output[output_class] = keep_output[output_class] + 1
            input_array = []
            document_index = []
            doc_start = 0
            doc_end = 0
            for g in keep_most_attentive_sentences:
                    input_array.append(keep_most_attentive_sentences[g])
            doc_end = doc_start + len(keep_most_attentive_sentences)
            # print(len(keep_most_attentive_sentences), files_path[elems][0])
            tup = (doc_start, doc_end)
            document_index.append(tup)
            del keep_most_attentive_sentences
            input_array = np.asarray(input_array)
            input_array = type_cast_float(input_array)
            test_result, attention_given = test_model(input_array, document_index)
            mean_abs_deviation, num_of_rights, lgth, clss = accuracy(test_result, class_label, num_of_class)
            sum_ = sum_+ mean_abs_deviation
            keep_num_of_rights = keep_num_of_rights + num_of_rights
#            final_class = 0
#            max_vote = 0
#            # print(keep_output ,files_path[elems])
#            for j in range(len(keep_output)):
#                if keep_output[j] > max_vote:
#                    max_vote = keep_output[j]
#                    final_class = j
#            if final_class is actual_class:
#                keep_num_of_rights = keep_num_of_rights + 1
#            else:
#                mean_abs_deviation = abs(final_class - actual_class)
#                sum_ = sum_ + mean_abs_deviation
    print("epoch - ", epoch)
    print("Avg mean on validation - ", sum_/count)
    print("Avg accuracy on validation - ", keep_num_of_rights/count * 100)
    if min_avg_mean > sum_/count:
        min_avg_mean = sum_/count
        min_avg_mean_epoch = epoch
    if max_avg_acc < keep_num_of_rights/count * 100:
        max_avg_acc = keep_num_of_rights/count * 100
        max_avg_acc_epoch = epoch
    # print("num of files - ", count) 
    print("#######################################################################")
                # mean_abs_deviation, num_of_rights, lgth = accuracy(test_result, class_label, num_of_class)
print("Min deviation obtained ", min_avg_mean, " at epoch ", min_avg_mean_epoch)
print("Max accuracy obtained ", max_avg_acc, " at epoch ", max_avg_acc_epoch)
print("number of files - ", num_of_test_files)

