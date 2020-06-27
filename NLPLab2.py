#!/usr/bin/env python
# coding: utf-8

# # Lexicons
# ### Lexicon is collection of words/Phrases with Information such as POS, tense Definitions etc.
# ### Lexicon has lexical entries, each entry is Word/Phrase that has a HeadWord(a.k.a Lemma) and Information

# In[1]:


#1. Stopwords:
from nltk.corpus import stopwords
stopwords.words('english')


# In[2]:


#2. CMU WordList
import nltk
entries = nltk.corpus.cmudict.entries()
len(entries)


# In[3]:


entries[:100]


# In[4]:


#3. CMU Wornet
from nltk.corpus import wordnet as wn
wn.synsets('AMAZING')


# In[5]:


wn.synset('amazing.s.01').lemma_names()


# In[6]:


#Task 2 - SIMPLE TEXT CLASSIFIER
def gender_features(word):
    return{'last_letter':word[-1]}
gender_features('TRUMP')


# In[7]:


from nltk.corpus import names
labeled_names = ([(name, 'male') for name in names.words('male.txt')]+[(name, 'female') for name in names.words('female.txt')])


# In[8]:


import random
random.shuffle(labeled_names)


# In[9]:


featuresets = [(gender_features(n),gender) for (n,gender) in labeled_names]


# In[10]:


train_set, test_set = featuresets[500:],featuresets[:500]


# In[11]:


import nltk
classifier = nltk.NaiveBayesClassifier.train(train_set)


# In[12]:


classifier.classify(gender_features('David'))


# In[13]:


classifier.classify(gender_features('Trump'))


# In[14]:


classifier.classify(gender_features('Michele'))


# In[15]:


print(nltk.classify.accuracy(classifier, test_set))


# In[16]:


# Tool names for positive and negative
#TEXTBLOB
#Jensen
# CMU Sphinx
#Task3: Vectorizers & Cosine Similarity
# CountVectorizer - Learn
# TfidfVectorizer
# from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


# In[17]:


# fit method vectorizes all the unique words in the corpus
vect = CountVectorizer(binary = True)
corpus = ["Tessaract is good optical character recognition engine  ", "optical character recognition is significant "]
vect.fit(corpus)


# In[18]:


vocab = vect.vocabulary_


# In[19]:


for key in sorted(vocab.keys()):
    print("{}:{}".format(key,vocab[key]))


# In[20]:


# do plagarism task using vectorization
print(vect.transform(["This is a good optical illusion"]).toarray())


# In[21]:


print(vect.transform(corpus).toarray())


# In[22]:


from sklearn.metrics.pairwise import cosine_similarity


# In[23]:


similarity = cosine_similarity(vect.transform(["Google Cloud Vision is a character recognition engine"]).toarray(), vect.transform(["OCR is an optical character recognition engine"]).toarray())


# In[24]:


print(similarity[0][0])


# In[ ]:




