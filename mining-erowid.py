
# coding: utf-8

## Text Mining the Erowid Experience Vault

##### import packages and set path

# In[1]:

import os, sys, re, math
path = os.path.abspath("")+"/"
print path


#### generator to iterate files into memory as lists of words

# In[2]:

import nltk, nltk.tokenize, nltk.corpus
from bs4 import BeautifulSoup

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def is_okay_word(s):
    import re
    if len(s)==1:
        return False
    elif is_number(s) and float(s)<1900:
        return False
    elif re.match('\d+[mM]?[gGlLxX]',s):
        return False
    elif re.match('\d+[oO][zZ]',s):
        return False
    else:
        return True

def yield_body_text(path):
    stopwords = nltk.corpus.stopwords.words('english')
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    tagger = nltk.tag.UnigramTagger(nltk.corpus.brown.tagged_sents())
    lemmatizer = nltk.WordNetLemmatizer()
    from nltk.corpus import wordnet
    experiences = os.listdir(path+'xml')
    for n, experience in enumerate(experiences):
        #if n<9900:
            #continue
        #if n>10000:
            #return
        words = []
        with open(path+"xml/"+experience) as file:
            soup = BeautifulSoup(file)
            #print experience
            tokens = tokenizer.tokenize(soup.bodytext.contents[0])
            pos = tagger.tag(tokens)
            words = []
            for token in pos:
                if token[1] == 'NN':
                    pos = wordnet.NOUN
                elif token[1] == 'JJ':
                    pos = wordnet.ADJ
                elif token[1] == 'VB':
                    pos = wordnet.VERB
                elif token[1] == 'RV':
                    pos = wordnet.ADV
                else:
                    pos = wordnet.NOUN
                lemma = lemmatizer.lemmatize(token[0], pos)
                if is_okay_word(lemma) and lemma not in stopwords:
                    words.append(lemma)
        if n%1000==0:
            print("Finished " + str(n) + " files out of " + str(len(experiences)))
        yield " ".join(words)


### create the bag-of-words using CountVectorizer

# In[3]:

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(min_df=2)
bag = cv.fit_transform(yield_body_text(path))


##### Save or load the serialized matrix

# In[4]:

import pickle
pickle.dump(bag, open("bag.p","wb"))
#bag = pickle.load(open("bag.p","rb"))


#### Determine feature weights

# In[103]:

from sklearn.feature_extraction.text import TfidfTransformer
tf = TfidfTransformer()
tfidf_bag = tf.fit_transform(bag)


#### Model topics

# In[149]:

from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=100)
svd_bag = svd.fit_transform(tfidf_bag)


##### Get lists of important terms for each topic

# In[153]:

def get_topic_names(model, vzr):
    topics = []
    for i in range(np.shape(model.components_)[0]):
        weights = list(model.components_[i])
        z = zip(weights,vzr.get_feature_names())
        z.sort(reverse=True)
        unzip = zip(*z)
        topics.append(unzip[1])
    return topics

topics = get_topic_names(svd,cv)

    

