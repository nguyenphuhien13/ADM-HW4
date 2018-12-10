from __future__ import division

import pickle
import math
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize


with open('totalcount.pkl', 'rb') as f:
    totalcount = pickle.load(f)
with open('dictionary.pkl', 'rb') as f:
    dictionary = pickle.load(f) # with index
myset = ['leggi','facebook','http','i','youtube','https','mq']

def preprocess(text):
    #--------------------------------------------------
    # preprocess the string/word
    #--------------------------------------------------
    
    string = ''
    stop_words = set(stopwords.words('italian'))
    words = text.strip().split()
    lemma = nltk.wordnet.WordNetLemmatizer() #lemmatization
    for r in words:
        if not r in stop_words:
            r = nltk.word_tokenize(r)
            r = [word.lower() for word in r if word.isalpha()]   
            r = ''.join(r) # convert r (list) back to string for lemmatization and concatenation
            if r not in myset:
                r = lemma.lemmatize(r)
                string = string + ' ' + r 
    return string

def extractlink(soup):
    #--------------------------------------------------
    # get link of a announcement
    #--------------------------------------------------
    
    links = []   
    for link in soup.findAll('a'):
        links.append(link.get('href'))
    return links

def index_one_file(termlist):
    #--------------------------------------------------
    #input = [word1, word2, ...]
    #output = {word1: [pos1, pos2], word2: [pos1, pos2], ...}
    #--------------------------------------------------
    
    fileIndex = {}
    words = list(set(termlist))
    word_list = [x for x in termlist]
    for i in range(len(word_list)):
        for item in words:
            if item == word_list[i]:
                fileIndex.setdefault(item, []).append(i)
    return fileIndex


def make_indices(dictionary):
    #--------------------------------------------------
    #input = {filename: [word1, word2, ...], ...}
    #ouput = {filename: {word: [pos1, pos2, ...]}, ...}
    #--------------------------------------------------
    
    total = {}
    for filename in dictionary.keys():
        new = dictionary[filename]
        total[filename] = index_one_file(new)
    return total


def fullIndex(regdex):
    #--------------------------------------------------
    # Dict reversal
    # input = {filename: {word: [pos1, pos2, ...], ... }} (a dictionary)
    # output = {word: {filename: [pos1, pos2]}, ...}, ...}t
    #--------------------------------------------------
    
    total_index = {}
    for filename in regdex.keys():
        for word in regdex[filename].keys():
            if word in total_index.keys():
                if filename in total_index[word].keys():
                    total_index[word][filename].extend(regdex[filename][word][:])
                else:
                    total_index[word][filename] = regdex[filename][word]
            else:
                total_index[word] = {filename: regdex[filename][word]}
    return total_index


def total_count(announcement):
    #--------------------------------------------------
    # Total word count
    # Count the number of words in a given announcement
    #--------------------------------------------------
    if announcement in totalcount.keys():
        value = totalcount.get(announcement)
    return value


def contain_count(term):
    #--------------------------------------------------
    # Contain count
    # Count the number of annoucements that contain a given term/word
    #--------------------------------------------------
    if term in dictionary.keys():
        return len(dictionary[term].keys()) 
    else:
        return 0

# TF-IDF
#--------------------------------------------------
# Calculate the TF-IDF score
#--------------------------------------------------
def tf(term_count, total_count):
    return term_count / total_count

def idf(doc_count, contain_count):
    return math.log(doc_count / contain_count)

def tf_idf(term_count, total_count, doc_count, contain_count):  
    return round(tf(term_count, total_count) * idf(doc_count, contain_count),2)


def jaccard(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return len(s1.intersection(s2)) / len(s1.union(s2))  
