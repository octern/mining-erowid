if "True":
	mypath = "/home/glenn/Desktop/brainclouds/"
	archive = mypath + "e/http80/www.erowid.org/experiences/"
	import sys, re, os
	sys.path.append(mypath)
	from urllib2 import urlopen
	from bs4 import BeautifulSoup, NavigableString
	from collections import defaultdict
	erowid = 'https://www.erowid.org'
	import math

class Vault:
	def __init__(self):
		from collections import defaultdict
		self.words = defaultdict(int)
		self.substances = {}
		self.tags = {}
		self.alldocs = 0
		self.ndocs = defaultdict(int)
		self.xtabs = defaultdict(int)
		from nltk.corpus import stopwords
		self.globalstops = stopwords.words('english')
	def tally(self, word, substances, tags):
		self.words[word]+=1
		for substance in substances:
			if substance not in self.substances:
				self.substances[substance] = defaultdict(int)
			self.substances[substance][word]+=1
		for tag in tags:
			if tag not in self.tags:
				self.tags[tag] = defaultdict(int)
			self.tags[tag][word]+=1
	def xtab(self, one, two):
		if one not in self.xtabs:
			self.xtabs[one] = defaultdict(int)
		if two not in self.xtabs:
			self.xtabs[two] = defaultdict(int)
		self.xtabs[one][two] += 1
		self.xtabs[two][one] += 1

	def scorewords(self,term):
		totalwords = 0
		scores = []
		total = 0
		for word in self.words:
			totalwords += self.words[word]
		if term in self.tags:
			freqs = self.tags
		elif term in self.substances:
			freqs = self.substances
		else:
			print "Bad search term."
			return
		for word in freqs[term]:
			if len(word)==1:
				continue
			if self.is_number(word) and float(word)<1900:
				continue
			if re.match('\d+[mM]?[gGlLxX]',word):
				continue
			if re.match('\d+[oO][zZ]',word):
				continue
			if word in self.stopwords[term]:
				continue
			if word in self.globalstops:
				continue
			score = (float(freqs[term][word])/self.ndocs[term])*math.log(float(self.alldocs)/self.words[word])
			scores.append((score,word))
		scores.sort(reverse=True)
		return scores
	def scorecat(self,term):
		scores = []
		if term not in self.xtabs:
			print "Bad search term."
			return
		for t in self.xtabs[term]:
			if self.xtabs[term][t]>=5:
				score = float(self.xtabs[term][t]*self.alldocs)/(self.ndocs[term]*self.ndocs[t])
				scores.append((score,t))
		scores.sort(reverse=True)
		return scores
	def parse(self,path):
		import nltk, nltk.tokenize
		tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
		lemmatizer = nltk.WordNetLemmatizer()
		all_xml = os.listdir(path+'xml');
		experiences = [filename for filename in all_xml if filename not in('tags.xml','badfiles.txt','stopwords.xml')]
		for n, experience in enumerate(experiences):
			#if n>100:
				#return
			file = open(path+'xml/'+experience)
			soup = BeautifulSoup(file)
			substances = [unicode(substance.contents[0]) for substance in soup.find_all("substance")]
			tags = [unicode(tag.contents[0]) for tag in soup.find_all("tag")]
			tokens = tokenizer.tokenize(soup.bodytext.contents[0])
			lemmas = set([lemmatizer.lemmatize(token.lower()) for token in tokens])
			for tag in tags:
				self.ndocs[tag]+=1
			for substance in substances:
				self.ndocs[substance]+=1
				for tag in tags:
					self.xtab(substance,tag)
					self.xtab(tag,substance)
			for word in lemmas:
				self.tally(word,substances, tags)
			if n%100==0:
				print("Finished " + str(n) + " files out of " + str(len(experiences)))
		self.alldocs = len(experiences)
		print "Finished all files."
	def csvwords(self, path):
		import csv
		with open(path + 'stopwords.csv','wb') as csvfile:
			writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
			for substance in self.substances:
				row = [substance,self.ndocs[substance]]
				score = self.scorewords(substance)[0:100]
				common = [t[1] for t in score]
				row += common
				if len(common)>0:
					writer.writerow(row)
	def get_stopwords(self,path):
		from collections import defaultdict
		file = open(path)
		soup = BeautifulSoup(file)
		self.stopwords = defaultdict(list)
		subtags = soup.find_all("substance")
		for subtag in subtags:
			substance = subtag.get("subname")
			stopwords = subtag.find_all("stopword")
			for stopword in stopwords:
				self.stopwords[substance].append(stopword.contents[0])


def doctext(path):
		import nltk, nltk.tokenize
		tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
		lemmatizer = nltk.WordNetLemmatizer()
		all_xml = os.listdir(path+'xml');
		experiences = [filename for filename in all_xml if re.match('\d+.xml',filename)]
		for n, experience in enumerate(experiences):
			#if n>100:
				#return
			with open(path+'xml/'+experience) as file:
				soup = BeautifulSoup(file)
				text = soup.bodytext.contents[0]
			if n%100==0:
				print("Finished " + str(n) + " files out of " + str(len(experiences)))
			yield text

def is_number(s):
	try:
		float(s)
		return True
	except ValueError:
		return False
		
def is_okay_word(word):
	import re
	if len(word)==1:
		return False
	elif is_number(word) and float(word)<1900:
		return False
	elif re.match('\d+[mM]?[gGlLxX]',word):
		return False
	elif re.match('\d+[oO][zZ]',word):
		return False
	else:
		return True

def doctext(path):
		from nltk.corpus import stopwords
		import nltk, nltk.tokenize
		tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
		lemmatizer = nltk.WordNetLemmatizer()
		stopw = stopwords.words('english')
		all_xml = os.listdir(path+'xml');
		experiences = [filename for filename in all_xml if re.match('\d+.xml',filename)]
		for n, experience in enumerate(experiences):
			#if n>10:
				#return
			words = []
			with open(path+'xml/'+experience) as file:
				soup = BeautifulSoup(file)
				tokens = tokenizer.tokenize(soup.bodytext.contents[0])
				lemmas = [lemmatizer.lemmatize(token.lower()) for token in tokens]
				for lemma in lemmas:
					if is_okay_word(lemma) and lemma not in stopw:
						words.append(lemma)	
			if n%100==0:
				print("Finished " + str(n) + " files out of " + str(len(experiences)))
			yield words

if True:
	from gensim import corpora, models, similarities
	mypath = "/home/glenn/Desktop/brainclouds/"
	dictionary = corpora.Dictionary()
	corpus = [dictionary.doc2bow(text, allow_update=True) for text in doctext(mypath)]
	dictionary.save(mypath+'bc_dict.dict')
	corpora.MmCorpus.serialize(mypath+'bc_corpus.mm', corpus)
	
if True:
	from gensim import corpora, models, similarities
	mypath = "/home/glenn/Desktop/brainclouds/"
	corpus = corpora.MmCorpus(mypath+'bc_corpus.mm')
	dictionary = corpora.Dictionary.load(mypath+'bc_dict.dict')
	
if True:
	tfidf = models.TfidfModel(corpus)
	corpus_tfidf = tfidf[corpus]

if True:
	lsi= models.ldamodel.LdaModel(corpus_tfidf, num_topics=100, id2word=dictionary)
	corpus_lsi = lsi[corpus_tfidf]

if True:
	lsi10 = models.ldamodel.LdaModel(corpus_tfidf, num_topics=10, id2word=dictionary)
	corpus_lsi10 = lsi10[corpus_tfidf]
	
if True:
	lda = models.ldamodel.LdaModel(corpus_tfidf, num_topics=100, id2word=dictionary)
	corpus_lda = lda[corpus_tfidf]
	
if True:
	hdp = models.hdpmodel.HdpModel(corpus_tfidf, id2word=dictionary)
	corpus_hdp = hdp(corpus_tfidf)
	
if True:
	corpora.MmCorpus.serialize(mypath+'bc_tfidf.mm', corpus_tfidf)
	corpora.MmCorpus.serialize(mypath+'bc_lsi.mm', corpus_lsi)
	lsi10.save(mypath+'model.lsi10')
	lsi.save(mypath+'model.lsi')
	tfidf.save(mypath + 'model.tfidf')
	
	
	from gensim import corpora, models, similarities
	mypath = "/home/glenn/Desktop/brainclouds/"
	corpus = corpora.MmCorpus(mypath+'bc_corpus.mm')
	dictionary = corpora.Dictionary.load(mypath+'bc_dict.dict')+'model.tfidf')
	
if True:
	from gensim import corpora, models, similarities
	mypath = "/home/glenn/Desktop/brainclouds/"
	corpus = corpora.MmCorpus(mypath+'bc_corpus.mm')
	dictionary = corpora.Dictionary.load(mypath+'bc_dict.dict')
	corpus_tdif = corpora.MmCorpus(mypath+'bc_corpus.mm')
	tfidf = models.TfidfModel.load(mypath + 'model.tfidf')
	lsi = models.LsiModel.load(mypath+'model.lsi')

if True:
	import gensim.matutils
	corpus_matrix = gensim.matutils.corpus2csc(corpus).T
	tfidf_matrix = gensim.matutils.corpus2csc(corpus_tfidf).T
	lsi_matrix = gensim.matutils.corpus2csc(corpus_lsi).T
	
if True:
	from sklearn.cluster import DBSCAN
	from sklearn.preprocessing import StandardScaler
	lsi_scaled = StandardScaler(with_mean=False).fit_transform(lsi_matrix)
	db = DBSCAN().fit(lsi_scaled)
	core_samples = db.core_sample_indices
	labels = db.labels_