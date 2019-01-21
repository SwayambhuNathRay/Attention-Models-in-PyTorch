
import torch
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
import math
import nltk
import collections
import re
import random
import pickle
from gensim.models.keyedvectors import KeyedVectors
import fnmatch
import codecs
import multiprocessing
from sklearn import metrics
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from gensim import models
from nltk.corpus import stopwords

max_sentence_length = 200
max_document_length = 100
frequency_cut_off = 100000
files_per_container = 400

folder = "/scratchd/home/swayambhu/Time_Stamping/APW/"
directory = "/scratchd/home/swayambhu/Time_Stamping/Dumped_files/APW_all_years/"
test_directory = "/scratchd/home/swayambhu/Time_Stamping/Dumped_files/APW_all_years/Test_Documents/"


# In[33]:


def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

def getWords(text):
    return re.compile('\w+').findall(text)

class DocumentContainer(object):
    def __init__(self, sentences, label, document_length):
        self.sentences = sentences
        self.label = label
        self.document_length = document_length


# In[34]:


def removeLowFreqWord(vocabulary, frequency):
    newDictionary = sorted(vocabulary.items(), key=lambda item: item[1])
    final_dictionary = dict(newDictionary[-frequency:])
    return final_dictionary

def indexing(vocabulary):
    k = 1
    for term in vocabulary:
        vocabulary[term] = k
        k = k + 1;
    return vocabulary


# In[35]:


def word2vec_glove(vocabulary):
    w2v_glove = pickle.load(open(directory+'w2v_glove.p',"rb"))
    word2vec_array = []
    word2vec_array.append(np.zeros(100).reshape(1,-1))
    for term in vocabulary:
        if term in w2v_glove:
            k = (w2v_glove[term]).reshape(-1,1)
            k = k.transpose()
            word2vec_array.append(k)
        else:
            random_array = np.random.randn(1,100).reshape(1,-1)
            word2vec_array.append(random_array)
    word2vec_array = np.asarray(word2vec_array)
    return word2vec_array


# # Building Vocabulary 

# In[36]:


def buildVocabulary(keep_file_path):
    vocabulary = collections.defaultdict(float)
    keep_train_file_path = []
    keep_max_sentence_length = 0
    keep_max_document_length = 0
    count = 0
    total_files = len(keep_file_path)
    train_files = int(0.8*total_files)
    for i in range(train_files):
        content = ""
        get_path = keep_file_path[i][0]
        get_class = keep_file_path[i][1]
        tup = (get_path, get_class)
        keep_train_file_path.append(tup)
        with open(get_path, "r", encoding='utf-8', errors='ignore') as inputFile:
            for lines in inputFile:
                sent_tokenize_list = sent_tokenize(lines)
                if len(sent_tokenize_list) > max_document_length:
                    keep_max_document_length = max_document_length
                elif len(sent_tokenize_list) > keep_max_document_length:
                    keep_max_document_length = len(sent_tokenize_list)
                for line in sent_tokenize_list:
                    line = line.lower().strip()
                    words = word_tokenize(line)
                    uniqueWords = set(wo for wo in words if (len(wo) > 2))
                    filtered_words = [word for word in uniqueWords if word not in stopwords.words('english')]
                    if len(filtered_words) > max_sentence_length:
                        keep_max_sentence_length = max_sentence_length
                    elif keep_max_sentence_length < len(filtered_words):
                        keep_max_sentence_length = len(filtered_words)
                    for word in filtered_words:
                        if word not in vocabulary:
                            vocabulary[word]=1
                        else:
                            vocabulary[word]+=1
    return vocabulary, keep_max_sentence_length, keep_max_document_length, keep_train_file_path, train_files


# # Building Container

# In[37]:


def buildContainer(get_container, vocabulary, word2vec_array, files_per_container, keep_max_sentence_length, keep_max_document_length):
    save_as_list_of_DocumentContainer = []
    num_of_files = 0
    for i in range(files_per_container):
        num_of_files = num_of_files+1
        get_path = get_container[i][0]
        with open(get_path, "r", encoding='utf-8', errors='ignore') as inputFile:
            new_array = np.zeros((keep_max_document_length, keep_max_sentence_length))
            for lines in inputFile:
                sent_tokenize_list = sent_tokenize(lines)
                if len(sent_tokenize_list) > keep_max_document_length:
                    document_size = keep_max_document_length
                else:
                    document_size = len(sent_tokenize_list)
                for index,line in enumerate(sent_tokenize_list):
                    line = line.lower().strip()
                    words = word_tokenize(line)
                    uniqueWords = set(wo for wo in words if (len(wo) > 2))
                    filtered_words = [word for word in uniqueWords if word not in stopwords.words('english')]
                    offset = 0
                    for word in filtered_words:
                        if word in vocabulary and index < keep_max_document_length and offset < keep_max_sentence_length:
                            new_array[index][offset] = vocabulary[word]
                            offset = offset + 1
        new_array = np.asarray(new_array)
        x = DocumentContainer(new_array, get_container[i][1], document_size)
        save_as_list_of_DocumentContainer.append(x)
    return save_as_list_of_DocumentContainer, num_of_files


# In[38]:


files_to_be_considered = 2
keep_all_file_path = []
print("Program Started.....")
for root, dirs, end in os.walk(folder):
        for file in end:
            if file.startswith('.'):
                continue
#             if files_to_be_considered is 0:
#                 files_to_be_considered = 2
#                 break
            files_to_be_considered = files_to_be_considered - 1
            get_path = os.path.join(root, file)
            save_year = int(file.split('.')[0].split('_')[2][0:4]) - 2000
            save_month = int(file.split('.')[0].split('_')[2][4:6])
            tup = (get_path, save_year*12 + save_month)
            keep_all_file_path.append(tup)
#             flag = 0
#             with open(get_path, "r", encoding='utf-8', errors='ignore') as inputFile:
#                 for lines in inputFile:
#                     sent_tok = sent_tokenize(lines)
#                     for line in sent_tok:
#                         word_tok = word_tokenize(line)
#                         uniqueWords = set(words for words in word_tok if len(words) > 2)
#                         filtered_words = [word for word in uniqueWords if word not in stopwords.words('english')]
#                         if len(filtered_words) > max_sentence_length:
#                             flag = 1
#                             break
#             if flag is 0:
                
random.shuffle(keep_all_file_path)
pickle.dump(keep_all_file_path, open(directory+"keep_all_file_path.p", "wb"))
print("All file path list dumped with ", len(keep_all_file_path), " files")
vocabulary, keep_max_sentence_length, keep_max_document_length, keep_train_files_path, train_files = buildVocabulary(keep_all_file_path)
print("Max sentence length - ", keep_max_sentence_length)
print("Max Document Length - ", keep_max_document_length)
vocabulary = removeLowFreqWord(vocabulary, frequency_cut_off)
vocabulary = indexing(vocabulary)
word2vec_array = word2vec_glove(vocabulary)
word2vec_array = np.squeeze(word2vec_array, axis=1)
print("vocabulary & word 2 vec array created")
print("Length of Vocabulary - ", len(vocabulary))
print("Shape of w2v array - ", word2vec_array.shape)
print("Number of train files - ", len(keep_train_files_path))
num_of_container = math.ceil(len(keep_train_files_path)/files_per_container)
balance = num_of_container*files_per_container - len(keep_train_files_path)
for i in range(balance):
    keep_train_files_path.append(keep_train_files_path[i])
for i in range(num_of_container):
    get_container = keep_train_files_path[i*files_per_container:(i+1)*files_per_container]
    list_of_Documents, number_of_files = buildContainer(get_container, vocabulary, word2vec_array, files_per_container, keep_max_sentence_length, keep_max_document_length)
    pickle.dump(list_of_Documents, open(directory+"List_of_Documents/list_of_documents_"+str(i)+".p", "wb"))


# In[39]:


print("Container of documents created and dumped")
print("Number of container created - ", num_of_container, " with ", files_per_container," files per container")
pickle.dump(vocabulary, open(directory+"vocabulary.p", "wb"))
print("Vocabulary Dumped")
pickle.dump(word2vec_array, open(directory+"w2varray.p", "wb"))
print("word 2 vec array dumped")
print("Data Ready for ", len(keep_train_files_path), " files")


# # Dumping Test Files

# In[40]:


def testFilePath(keep_all_test_file_path, vocabulary, given_bucket):
    keep_file_path=[]
    for i in range(len(keep_all_test_file_path)):
        get_path = keep_all_test_file_path[i][0]
        get_class = keep_all_test_file_path[i][1]
        tup = (get_path, get_class)
        with open(get_path, "r", encoding = 'utf-8', errors = 'ignore') as inputFile:
            for lines in inputFile:
                tokenize = sent_tokenize(lines)
                sen = len(tokenize)
                if sen <= given_bucket:
                    keep_file_path.append(tup)
    return keep_file_path


# In[41]:


def buildTestContainer(keep_file_path, vocabulary, word2vec_array):
    save_as_list_of_DocumentContainer = []
    num_of_files = len(keep_file_path)
    for i in range(num_of_files):
        get_path = keep_file_path[i][0]
        with open(get_path, "r", encoding='utf-8', errors='ignore') as inputFile:
            new_array = np.zeros((keep_max_document_length, keep_max_sentence_length))
            for lines in inputFile:
                sent_tokenize_list = sent_tokenize(lines)
                if len(sent_tokenize_list) > keep_max_document_length:
                    document_size = keep_max_document_length
                else:
                    document_size = len(sent_tokenize_list)
                for index, line in enumerate(sent_tokenize_list):
                    content = line.lower().strip()
                    words = word_tokenize(content)
                    uniqueWords = set(wo for wo in words if (len(wo) > 2))
                    filtered_words = [word for word in uniqueWords if word not in stopwords.words('english')]
                    offset = 0
                    for word in filtered_words:
                        if word in vocabulary and offset < keep_max_sentence_length and index < keep_max_document_length:
                            new_array[index][offset] = vocabulary[word]
                            offset = offset + 1
        new_array = np.asarray(new_array)
        x = DocumentContainer(new_array, keep_file_path[i][1], document_size)
        save_as_list_of_DocumentContainer.append(x)
    return save_as_list_of_DocumentContainer, num_of_files, keep_file_path


# # Dumping development files

# In[42]:


keep_all_test_file_path = []
development_files = int(0.9*len(keep_all_file_path))
keep_all_test_file_path = keep_all_file_path[train_files:development_files]
vocabulary = pickle.load(open(directory+'vocabulary.p',"rb"))
w2varray = pickle.load(open(directory+'w2varray.p',"rb"))
w2varray = np.asarray(w2varray)
arr = [5, 10, 20, 30, 50, 100, 2000]
for j in range(len(arr)):
    sentence_length = arr[j]
    keep_test_file_path = testFilePath(keep_all_test_file_path, vocabulary, sentence_length)
    pickle.dump(keep_test_file_path, open(test_directory + "development_file_path" + str(arr[j]) + ".p", "wb"))
    num_of_test_files = len(keep_test_file_path)
    print("Number of development files - ",num_of_test_files)
    list_of_Documents, number_of_files, keep_test_file_path = buildTestContainer(keep_test_file_path, vocabulary, w2varray)
    pickle.dump(list_of_Documents, open(test_directory+"list_of_development_documents" + str(sentence_length) +".p", "wb"))


# # Dumping test files and test + development files 

# In[44]:


keep_all_test_file_path = keep_all_file_path[development_files:len(keep_all_file_path)]
for j in range(len(arr)):
    sentence_length = arr[j]
    keep_test_file_path = testFilePath(keep_all_test_file_path, vocabulary, sentence_length)
    pickle.dump(keep_test_file_path, open(test_directory + "test_file_path" + str(arr[j]) + ".p", "wb"))
    num_of_test_files = len(keep_test_file_path)
    print("Number of test files - ",num_of_test_files)
    list_of_Documents, number_of_files, keep_test_file_path = buildTestContainer(keep_test_file_path, vocabulary, w2varray)
    pickle.dump(list_of_Documents, open(test_directory+"list_of_test_documents" + str(sentence_length) +".p", "wb"))


for i in range(len(arr)):
    keep_dev_file = []
    keep_dev_file = pickle.load(open(test_directory + "list_of_development_documents" + str(arr[i]) + ".p", "rb"))
    keep_dev_file_path = pickle.load(open(test_directory + "development_file_path" + str(arr[i]) + ".p", "rb"))
    keep_test_file = []
    keep_test_file = pickle.load(open(test_directory + "list_of_test_documents" + str(arr[i]) + ".p", "rb"))
    keep_test_file_path = pickle.load(open(test_directory + "test_file_path" + str(arr[i]) + ".p", "rb"))
    keep_final_path = []
    keep_final_path = keep_dev_file + keep_test_file
    keep_all_test_file_path = keep_dev_file_path + keep_test_file_path
    pickle.dump(keep_final_path, open(test_directory + "list_of_full_test_documents"+str(arr[i])+".p", "wb"))
    print("Full test files dumped with ", len(keep_all_test_file_path), " files for "+ str(arr[i]) +" files")
    pickle.dump(keep_all_test_file_path, open(test_directory + "all_test_file_path"+str(arr[i])+".p", "wb"))


# In[ ]:





# In[ ]:




