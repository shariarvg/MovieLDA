# Text Cleaning

import operator
import numpy
import spacy
import csv

spacy.load('en')

#Load Spacy
from spacy.lang.en import English
parser = English()
def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens

#Using WordNet
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma
    
from nltk.stem.wordnet import WordNetLemmatizer
def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)

#Filter out stopwords
nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))

#Prepare text for an LDA
def prepare_text_for_lda(text):
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    return tokens

#Import and convert text data
import random
text_data = []
startname = "all_articles/"
journal = []
disciplines = {"anthropology":["american ethologist","current anthropolgy"], \
               "economics":["american economic review","econometrica","journal economic perspectives","journal finance",\
                            "quarterly journal economics","review economics statistics","review financial studies"], \
               "political science":["american journal political science","american political science review","british journal political science","international organization","journal politics","world politics"], \
               "psychology":["child development","psychological science"], \
               "sociology":["american journal sociology","american sociological review","demography","european sociological review","journal marriage family"]
               }

start = []
count = 0


#feed in articles
with open('citations.csv') as csv_file:
    csv_reader = csv.reader(csv_file)
    csv_reader = sorted(csv_reader,key=operator.itemgetter(441)) #for getting journal titles
    csvr = sorted(csv_reader,key=operator.itemgetter(441)) #for feeding into the model
    listcsv = list(csv_reader)

    for line in listcsv:
        if (line[0]=="filename_temp"):
           print("top row--------------")
        else:
            if line[441] not in journal:
                print(line[441])
                journal.append(line[441])
                start.append(count)
        count+=1
        
    for line in listcsv:
        
        if (line[0]!="filename_temp"):
            endname = line[0]+".txt"
            combined = startname + endname
            #print(combined)
            r = open(combined, 'r')
            rtext= r.read()
            if len(rtext)>1000000:
                print(endname)
            else:
                tokens = prepare_text_for_lda(rtext)
                text_data.append(tokens)


#Make a dictionary out of the words and convert it to a bag of words
from gensim import corpora
dictionary = corpora.Dictionary(text_data)
corpus = [dictionary.doc2bow(text) for text in text_data]
import pickle
pickle.dump(corpus, open('corpus.pkl', 'wb'))
dictionary.save('dictionary.gensim')

#import gensim, create and save model
import gensim
ntopics = 10
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = ntopics, id2word=dictionary, passes=15)
ldamodel.save('model5.gensim')

#Collect probability distributions
topicArr = []
for ele in text_data:
    bow = dictionary.doc2bow(ele)
    tops = ldamodel.get_document_topics(bow, minimum_probability=None, minimum_phi_value=None, per_word_topics=False)
    topicsCovered = []
    for ele in tops:
        topicsCovered.append(ele[0])
    for i in range(ntopics):
        if i not in topicsCovered:
            tops.append((i,0))
    topicArr.append(tops)

#Checkpoint for topic array
#print("Length of Topic Array: " + str(len(topicArr)))
#print(topicArr[0])

start.append(28986)
#markers = [19,34,62,63,407,509,628,1434,1572,1741,1750,1752,1803,1916,2058,2150,2244,2248,2255]
markers = start[1:]
endpoints = start
discArr = [[0 for x in range(ntopics)] for y in range(5)]


#Aggregate across disciplines
discValues = list(disciplines.values())
discTotal = [0,0,0,0,0] #number of articles in each discipline

for count in range(len(topicArr)):
    for i in range(22): #number of journals
        if count<markers[i]:
            for p in range(5):
                if (journal[i] in discValues[p]):
                    for go in range(ntopics):
                        topid = topicArr[count][go][0]
                        discArr[p][topid]+=topicArr[count][go][1]
                        discTotal[p]+=1
            break
        
#Divide by number of films in genre
for x in range(len(discArr)):
    for y in range(ntopics):
        discArr[x][y]/=discTotal[p]

#Checkpoint for journal array
print(discArr)

dists = []
for af in range(len(topicArr)):
    dists.append(0)
count = 0
i = 0
go = 0

match = [] #holds genre for each movie index
#finding average distance for each movie


#Calculate Distances
for count in range(len(topicArr)):
    for i in range(22):#number of 
        if count<markers[i]:
            for go in range(ntopics):
                curr = journal[i]
                topid = topicArr[count][go][0]
                for p in range(5):
                    if (curr in discValues[p]):
                        disc = discArr[p]
                        break
                dists[count]+=1-(abs(topicArr[count][go][1]-disc[topid])/(topicArr[count][go][1]+disc[topid]+0.0001))
            dists[count]=1/(dists[count]+0.00001)
            break

print(discArr[0])
print("-----------------")
for ele in dists:
    print(ele)
