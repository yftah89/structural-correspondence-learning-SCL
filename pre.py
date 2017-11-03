import xml.etree.ElementTree as ET
import random
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mutual_info_score
import pickle
import os

def XML2arrayRAW(neg_path, pos_path):
    reviews = []
    negReviews = []
    posReviews = []

    neg_tree = ET.parse(neg_path)
    neg_root = neg_tree.getroot()
    for rev in neg_root.iter('review'):
        reviews.append(rev.text)
        negReviews.append(rev.text)



    pos_tree = ET.parse(pos_path)
    pos_root = pos_tree.getroot()

    for rev in pos_root.iter('review'):
        reviews.append(rev.text)
        posReviews.append(rev.text)

    return reviews,negReviews,posReviews



def split_data_balanced(reviews,dataSize,testSize):
    test_data_neg = random.sample(range(0, dataSize), testSize)
    test_data_pos = random.sample(range(dataSize, 2*dataSize), testSize)
    random_array = np.concatenate((test_data_neg, test_data_pos))
    train = []
    test = []
    test_target = []
    train_target = []
    for i in range(0, 2*dataSize):
        if i in random_array:
            test.append(reviews[i])
            target = 0 if i < dataSize else 1
            test_target.append(target)
        else:
            train.append(reviews[i])
            target = 0 if i < dataSize else 1
            train_target.append(target)
    return train, train_target, test, test_target

def extract_and_split(neg_path, pos_path):
    reviews,n,p = XML2arrayRAW(neg_path, pos_path)
    train, train_target, test, test_target = split_data_balanced(reviews,1000,200)
    return train, train_target, test, test_target


def GetTopNMI(n,CountVectorizer,X,target):
    MI = []
    length = X.shape[1]


    for i in range(length):
        temp=mutual_info_score(X[:, i], target)
        MI.append(temp)
    MIs = sorted(range(len(MI)), key=lambda i: MI[i])[-n:]
    return MIs,MI


def getCounts(X,i):

    return (sum(X[:,i]))

def preproc(pivot_num,pivot_min_st,src,dest):

    pivotsCounts= []
    unlabeled = []
    names = []

    filename = src + "_to_" + dest + "/split/"
    if not os.path.exists(os.path.dirname(filename)):
        #gets all the train and test for sentiment classification
        train, train_target, test, test_target = extract_and_split("data/"+src+"/negative.parsed","data/"+src+"/positive.parsed")
    else:
        with open(src + "_to_" + dest + "/split/train", 'rb') as f:
            train = pickle.load(f)
        with open(src + "_to_" + dest + "/split/test", 'rb') as f:
            test = pickle.load(f)
        with open(src + "_to_" + dest + "/split/train_target", 'rb') as f:
            train_target = pickle.load(f)
        with open(src + "_to_" + dest + "/split/test_target", 'rb') as f:
            test_target = pickle.load(f)


    # sets x train matrix for classification
    bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=5,binary=True)
    X_2_train = bigram_vectorizer.fit_transform(train).toarray()

    # gets all the train and test for pivot classification
    unlabeled,source,target=XML2arrayRAW("data/"+src+"/"+src+"UN.txt","data/"+dest+"/"+dest+"UN.txt")
    source=source+train
    src_count = 20
    un_count = 40


    unlabeled=source+target

    bigram_vectorizer_unlabeled = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=un_count, binary=True)
    X_2_train_unlabeled = bigram_vectorizer_unlabeled.fit_transform(unlabeled).toarray()


    bigram_vectorizer_source = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=src_count, binary=True)
    X_2_train_source = bigram_vectorizer_source .fit_transform(source).toarray()

    bigram_vectorizer_target = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=20, binary=True)
    X_2_train_target = bigram_vectorizer_target.fit_transform(target).toarray()

    MIsorted,RMI=GetTopNMI(2000,CountVectorizer,X_2_train,train_target)
    MIsorted.reverse()
    c=0
    i=0

    while (c<pivot_num):
        name= bigram_vectorizer.get_feature_names()[MIsorted[i]]

        s_count = getCounts(X_2_train_source,bigram_vectorizer_source.get_feature_names().index(name)) if name in bigram_vectorizer_source.get_feature_names() else 0
        t_count = getCounts(X_2_train_target, bigram_vectorizer_target.get_feature_names().index(name)) if name in bigram_vectorizer_target.get_feature_names() else 0
        if(s_count>=pivot_min_st and t_count>=pivot_min_st):
            names.append(name)
            pivotsCounts.append(bigram_vectorizer_unlabeled.get_feature_names().index(name))
            c+=1
         #   if(c<100):
              #  print "feature is ",name," it MI is ",RMI[MIsorted[i]]," in source ",s_count," in target ",t_count
        i+=1



    source_valid = len(source)/5
    target_valod = len(target)/5
    c=0
    y = X_2_train_unlabeled[:,pivotsCounts]
    print "old x is ", X_2_train_unlabeled.shape
    x =np.delete(X_2_train_unlabeled, pivotsCounts, 1)  # delete second column of C
    print "new x is ", x.shape
    #taking 1500 from books and 4000 from kitchen
    x_valid=np.concatenate((x[:source_valid][:], x[-target_valod:][:]), axis=0)

    x = x[source_valid:-target_valod][:]


    # taking 1500 from books and 4000 from kitchen
    y_valid = np.concatenate((y[:source_valid][:], y[-target_valod:][:]), axis=0)

    y = y[source_valid:-target_valod][:]
    filename = src+"_to_"+dest+"/"+"pivot_names/pivot_names_"+src+"_"+dest+"_"+str(pivot_num)+"_"+str(pivot_min_st)
    if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))


    with open(filename, 'wb') as f:
        pickle.dump(names, f)
    filename = src + "_to_" + dest + "/split/"
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
        with open(src + "_to_" + dest + "/split/train", 'wb') as f:
            pickle.dump(train, f)
        with open(src + "_to_" + dest + "/split/test", 'wb') as f:
            pickle.dump(test, f)
        with open(src + "_to_" + dest + "/split/train_target", 'wb') as f:
            pickle.dump(train_target, f)
        with open(src + "_to_" + dest + "/split/test_target", 'wb') as f:
            pickle.dump(test_target, f)
    filename = src+"_to_"+dest+"/"+"pivotsCounts/"+"pivotsCounts"+src+"_"+dest+"_"+str(pivot_num)+"_"+str(pivot_min_st)
    if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
    with open(src+"_to_"+dest+"/"+"pivotsCounts/"+"pivotsCounts"+src+"_"+dest+"_"+str(pivot_num)+"_"+str(pivot_min_st), 'wb') as f:
        pickle.dump(pivotsCounts, f)



    return x,y,x_valid,y_valid,x.shape[1]







