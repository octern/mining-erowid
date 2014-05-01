if True:
	path = "C:/Documents and Settings/M543015/Desktop/text/"
	import string
	documents = []
	for i in range(10):
		s = ""
		with open(path+str(i)+".txt") as f:
			content = f.readlines()
			joined = " ".join(content)
			fixed = string.replace(joined, '\n',' ')
			uni = unicode(fixed, errors='ignore')
			documents.append(uni)
				
if True:
	stoplist = set('for a of the and to in'.split())
	texts = [[word for word in document.lower().split() if word not in stoplist] for document in documents]
	all_tokens = sum(texts, [])
	tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
	texts = [[word for word in text if word not in tokens_once] for text in texts]   

if True:
	from gensim import corpora, models, similarities
	dictionary = corpora.Dictionary(texts)
	corpus = [dictionary.doc2bow(text) for text in texts]

if True:
	tfidf = models.TfidfModel(corpus)
	corpus_tfidf = tfidf[corpus]
	
if True:
	lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2)
	doc = "Where on earth is that missing airplane?"
	vec_bow = dictionary.doc2bow(doc.lower().split())
	vec_lsi = lsi[vec_bow]

if True:
	index = similarities.MatrixSimilarity(lsi[corpus]) 
	sims = index[vec_lsi]
	print(list(enumerate(sims)))
	
if True:
	hdp = models.HdpModel(corpus_tfidf, id2word=dictionary)
	vec_hdp = hdp[vec_bow]
	index = similarities.MatrixSimilarity(hdp[corpus]) 
	sims = index[vec_hdp]
	print(list(enumerate(sims)))
	
if True:
	from sklearn import datasets
	iris = datasets.load_iris()
	type(iris)
	type(iris.data) #numpy dense array
	
if True:
	import gensim
	matrix = gensim.matutils.corpus2csc(corpus_tfidf)
	tmat = matrix.T
	from sklearn import cluster, datasets
	k_means = cluster.KMeans(n_clusters=2)
	k_means.fit(tmat) 
	k_means.labels_
	
if True:
	k_means = cluster.KMeans(n_clusters=3)
	k_means.fit(tmat) 
	k_means.labels_

if True:
	real_y = [0,1,1,1,1,0,0,0,0,0]
	from sklearn.neighbors import KNeighborsClassifier
	knn = KNeighborsClassifier()
	knn.fit(tmat,real_y)
	knn.predict(tmat)

if True:
	docmat = gensim.matutils.corpus2csc([vec_bow])
	doct = docmat.T
	from scipy.sparse import csc_matrix, hstack
	n = tmat.shape[1] - doct.shape[1]
	zeros = csc_matrix((1,n))
	docr = hstack((doct, zeros))
	knn.predict(docr)
	
if True:
	doc = "I don't believe that the death penalty is appropriate for murder."
	vec_bow = dictionary.doc2bow(doc.lower().split())
	docmat = gensim.matutils.corpus2csc([vec_bow])
	doct = docmat.T
	from scipy.sparse import csc_matrix, hstack
	n = tmat.shape[1] - doct.shape[1]
	zeros = csc_matrix((1,n))
	docr = hstack((doct, zeros))
	knn.predict(docr)
	
if True:
	import gensim
	matrix = gensim.matutils.corpus2csc(corpus_lsi)
	tmat = matrix.T
	from sklearn import cluster, datasets
	k_means = cluster.KMeans(n_clusters=2)
	k_means.fit(tmat) 
	k_means.labels_
	doc = "I don't believe that the death penalty is appropriate for murder."
	vec_bow = dictionary.doc2bow(doc.lower().split())
	vec_lsi = lsi[vec_bow]
	docmat = gensim.matutils.corpus2csc([vec_lsi])
	doct = docmat.T
	from scipy.sparse import csc_matrix, hstack
	n = tmat.shape[1] - doct.shape[1]
	if n > 0:
		zeros = csc_matrix((1,n))
		docr = hstack((doct, zeros))
	else:
		docr = doct
	real_y = [0,1,1,1,1,0,0,0,0,0]
	from sklearn.neighbors import KNeighborsClassifier
	knn = KNeighborsClassifier()
	knn.fit(tmat,real_y)
	knn.predict(docr)

if True:
	doc = "Where on earth is that missing airplane?"
	vec_bow = dictionary.doc2bow(doc.lower().split())
	vec_lsi = lsi[vec_bow]
	docmat = gensim.matutils.corpus2csc([vec_lsi])
	doct = docmat.T
	from scipy.sparse import csc_matrix, hstack
	n = tmat.shape[1] - doct.shape[1]
	if n > 0:
		zeros = csc_matrix((1,n))
		docr = hstack((doct, zeros))
	else:
		docr = doct
	knn.predict(docr)