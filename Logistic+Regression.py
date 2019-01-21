
# coding: utf-8

# In[52]:


import os
import sys
import numpy as np
from scipy import sparse
import pickle
import multiprocessing
import re
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import ElasticNet
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import operator
import collections
import random


# In[53]:


# folder = "/scratchd/home/swayambhu/Text_Data/"


# In[54]:


def buildVocabulary(keep_train_file, get_grams):
    vocabulary = collections.defaultdict(float)
    index = 0
    uni_gram = []
    bi_gram = []
    tri_gram = []
    keep_file_name = []
    files_to_be_considered = 500
    for j in range(len(keep_train_file)):
        print("Building Vocabulary")
        # if files_to_be_considered is 0:
        #     files_to_be_considered = 500
        #     break
        files_to_be_considered = files_to_be_considered - 1
        file = keep_train_file[j]
        keep_file_name.append(file)
        for i in range(3):
            get_gram_path = get_grams + str(i+1) + "/" + file
            with open(get_gram_path, "r", encoding='utf-8', errors='ignore') as inputFile:
                for word in inputFile:
                    if len(word) >= 2:
                        if word not in vocabulary:
                            vocabulary[word] = index
                            if i is 0:
                                uni_gram.append(index)
                            elif i is 1:
                                bi_gram.append(index)
                            elif i is 2:
                                tri_gram.append(index)
                            index = index + 1

    print(index)
    return keep_file_name, vocabulary, uni_gram, bi_gram, tri_gram


# In[55]:


def generateFeatures(get_grams,regression, keep_file, vocabulary):
    num_of_files = len(keep_file)
    len_of_vocab = len(vocabulary)
    temp_feature_matrix = []
    class_label = []
    num = 0
    init = 0
    final_feature_matrix = sparse.csr_matrix([])
    for i in range(num_of_files):
        num = num + 1
        file = keep_file[i]
        save_year = int(file.split('.')[0].split('_')[2][0:4]) - 2000
        save_month = int(file.split('.')[0].split('_')[2][4:6])
        temp_array = [0]*len_of_vocab
        for j in range(3):
            get_file_path = get_grams + str(j+1) + "/" + file
            with open(get_file_path,  "r", encoding='utf-8', errors='ignore') as inputFile:
                for word in inputFile:
                    temp_array[vocabulary[word]] = 1
        temp_array.append(save_year*12 + save_month)
        temp_array = np.asarray(temp_array).reshape(1, -1)
        temp_feature_matrix.append(temp_array)
        if num%50 is 0:
            print("Converting")
            temp_feature_matrix = np.array(temp_feature_matrix).squeeze(1)
            print(temp_feature_matrix.shape)
            if init is 1:
#                 temp_feature_matrix = np.asarray(temp_feature_matrix).squeeze(1)
                tempFeatureVectorSparse = sparse.csr_matrix(np.array(temp_feature_matrix))
                final_feature_matrix = sparse.vstack([final_feature_matrix,tempFeatureVectorSparse])
                del tempFeatureVectorSparse
                del temp_feature_matrix
                temp_feature_matrix = []
            else:
                final_feature_matrix = sparse.csr_matrix(np.array(temp_feature_matrix))
                del temp_feature_matrix
                temp_feature_matrix = []
                init = 1
    pickle.dump(final_feature_matrix,open(regression+ "sparse_feature_matrix.p","wb"))
#     feature_matrix = np.asarray(feature_matrix).squeeze(1)
    print(final_feature_matrix.shape[0])


# # In[56]:


def trainRegression(regression):
    model = LogisticRegression(penalty='l2', max_iter=1000, verbose=1, n_jobs=30)
    # model = svm.SVC(C=5000.0, decision_function_shape='ovo', degree=3, kernel='rbf', max_iter=-1, probability=True, tol=0.001, verbose= 1);
    featureVectors = pickle.load(open(regression + "sparse_feature_matrix.p","rb"))
    featureVectors_csc = featureVectors.tocsc()
    trainX = featureVectors_csc[:,:-1].tocsr()
    print(trainX.shape)
    trainY = featureVectors_csc[:,-1].toarray()
#     print(trainY)
    classifier = OneVsRestClassifier(model)
    print("Training")
    classifier = classifier.fit(trainX,trainY)
    pickle.dump(classifier, open(regression + "regression_model.p","wb"))
    print("Done Training")


# # In[57]:


def testFilePath(keep_test_file):
    files_to_be_considered = 100
    keep_file_path = []
    for i in range(len(keep_test_file)):
        file = keep_test_file[i]
        # if files_to_be_considered is 0:
        #     files_to_be_considered = 100
        #     break
        files_to_be_considered = files_to_be_considered - 1
        keep_file_path.append(file)

    random.shuffle(keep_file_path)
    return keep_file_path


# In[58]:


def testFeatureGeneration(test_file, regression, get_test_grams, vocabulary):
    num_of_files = len(test_file)
    num = 0
    init = 0
    len_of_vocab = len(vocabulary)
    temp_feature_matrix = []
    final_feature_matrix = sparse.csr_matrix([])
    for i in range(num_of_files):
        num = num + 1
        file = test_file[i]
        save_year = int(file.split('.')[0].split('_')[2][0:4]) - 2000
        save_month = int(file.split('.')[0].split('_')[2][4:6])
        temp_array = [0]*len_of_vocab
        for j in range(3):
            get_file_path = get_test_grams + str(j+1) + "/" + file
            with open(get_file_path,  "r", encoding='utf-8', errors='ignore') as inputFile:
                for word in inputFile:
                    if word in vocabulary:
                        temp_array[vocabulary[word]] = 1
            
        temp_array.append(save_year*12 + save_month)
        temp_array = np.asarray(temp_array).reshape(1, -1)
        temp_feature_matrix.append(temp_array)
        if num%50 is 0:
            print("Converting")
            temp_feature_matrix = np.array(temp_feature_matrix).squeeze(1)
            print(temp_feature_matrix.shape)
            if init is 1:
#                 temp_feature_matrix = np.asarray(temp_feature_matrix).squeeze(1)
                tempFeatureVectorSparse = sparse.csr_matrix(np.array(temp_feature_matrix))
                final_feature_matrix = sparse.vstack([final_feature_matrix,tempFeatureVectorSparse])
                del tempFeatureVectorSparse
                del temp_feature_matrix
                temp_feature_matrix = []
            else:
                final_feature_matrix = sparse.csr_matrix(np.array(temp_feature_matrix))
                del temp_feature_matrix
                temp_feature_matrix = []
                init = 1
    pickle.dump(final_feature_matrix,open(regression+ "sparse_feature_test_matrix.p","wb"))
    print(final_feature_matrix.shape[0])


# In[67]:


def testRegression(regression, uni_gram, bi_gram, tri_gram):
    model = pickle.load(open(regression+"regression_model.p","rb"))
    print(model.coef_.shape)
    keep_weights = model.coef_
    unigram_weight = 0
    bigram_weight = 0
    trigram_weight = 0
    for i in range(len(keep_weights)):
        for j in range(len(uni_gram)):
            unigram_weight = unigram_weight + abs(keep_weights[i][uni_gram[j]])
        for j in range(len(bi_gram)):
            bigram_weight = bigram_weight + abs(keep_weights[i][bi_gram[j]])
        for j in range(len(tri_gram)):
            trigram_weight = trigram_weight + abs(keep_weights[i][tri_gram[j]])
        print(unigram_weight/len(uni_gram) , bigram_weight/len(bi_gram), trigram_weight/len(tri_gram))

    # print(keep_weights)
    featureVectors = pickle.load(open(regression + "sparse_feature_test_matrix.p","rb"))
    featureVectors_csc = featureVectors.tocsc()
    testX = featureVectors_csc[:,:-1].tocsr()
    # print(testX.shape)
    testY = featureVectors_csc[:,-1].toarray()
#     print(testY)
    predictions = model.predict(testX)
#     print(testY)
    # print(predictions, testY)
    dev = 0.0
    count = 0
    for i in range(len(predictions)):
        dev += abs(predictions[i] - testY[i])
#        print((predictions[i] - testY[i])[0])
        if (predictions[i] - testY[i])[0] == 0:
            count = count + 1
    avgDev = dev/len(predictions)
    avgPre = count/len(predictions)
    print("Average Deviation - ", avgDev)
    print("Accuracy - ", avgPre * 100, "%")


# In[68]:


def final():
    folder = "/scratchd/home/swayambhu/APW"
    get_grams = "/scratchd/home/swayambhu/N-grams/"
    directory = "/scratchd/home/swayambhu/Time_Stamping/"
    regression = "/scratchd/home/swayambhu/Regression/"
    # files_to_be_considered = 1
    keep_all_file_path = []
    uni_gram = []
    bi_gram = []
    tri_gram = []
    keep_all_file_path = pickle.load(open(directory+'keep_all_file_path.p',"rb"))
    train_files = int(0.8*len(keep_all_file_path))
    keep_train_file_path = keep_all_file_path[0:train_files]
    keep_test_file_path = pickle.load(open(directory+'all_test_file_path5.p',"rb"))
    print("Test file path loaded")
    #keep_all_file_path[train_files:len(keep_all_file_path)]
    # keep_train_file_path = pickle.load(open(directory+'keep_train_file_path.p',"rb"))
    # keep_test_file_path = pickle.load(open(directory+'keep_test_file_path.p',"rb"))
    keep_train_file = []
    keep_test_file = []
    # print(keep_all_file_path)
    for i in range(len(keep_train_file_path)):
        get_path = keep_train_file_path[i][0].split('/')[-1]
        keep_train_file.append(get_path)
    # print(keep_train_file)
    for i in range(len(keep_test_file_path)):
        get_path = keep_test_file_path[i][0].split('/')[-1]
        keep_test_file.append(get_path)
    # get_grams_test = "/scratchd/home/siddesh/DocumentTimestamp/Regression/Ngrams_test/APW_all/"
    # test_folder = "/scratchd/home/siddesh/DocumentTimestamp/DateNDocWiseText/LDC_2000/APW"
    keep_file, vocabulary, uni_gram, bi_gram, tri_gram = buildVocabulary(keep_train_file, get_grams)
    # generateFeatures(get_grams, regression, keep_file, vocabulary)
    # trainRegression(regression)
    get_test_files = testFilePath(keep_test_file)
    testFeatureGeneration(get_test_files, regression, get_grams, vocabulary)
    testRegression(regression, uni_gram, bi_gram, tri_gram)


# In[69]:


final()


# In[ ]:




