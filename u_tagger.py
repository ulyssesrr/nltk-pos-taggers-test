# vim:fileencoding=utf-8

import numpy as np

from scipy.sparse import csr_matrix, vstack, hstack

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score

import nltk

class UTagger:

	def __init__(self, train):
		#self.count_vect = CountVectorizer()
		self.n_features = 2**14
		self.vectorizer = HashingVectorizer(decode_error='ignore', n_features=self.n_features, norm=None, binary=True, dtype=np.int32, non_negative=True)
		self.clf = BernoulliNB()
		self.stemmer = nltk.stem.RSLPStemmer()
		X, Y = self.extract_tsents_features(train)
		#print(X[:10])
		#print(Y[:10])
		self.clf.fit(X, Y)
		
	def extract_sents_features(self, sents):
		X = csr_matrix((0, self.n_features), dtype=np.int32)
		for sent in sents:
			sx = self.extract_sent_features(sent)
			X = vstack(X, sx)
		return X
	
	def extract_sent_features(self, sentence):
		X = []
		for i, word in enumerate(sentence):
			arr = self.extract_word_features(word, i == 0)
			X += [arr]
		return X
		
	def extract_word_features(self, word, first_word=False):
		first_isupper = 1 if (word[0].isupper() and not first_word) else 0
		st = self.stemmer.stem(word)
		suffix = word[len(st):]
		feature_string = "%s %s" % (st, suffix)
		#print(feature_string)
		arr = self.vectorizer.transform([feature_string])
		b = csr_matrix([[first_isupper]], dtype=np.int32)
		arr = hstack([b, arr], dtype=np.int32)
		#arr = np.insert(arr, 0, first_isupper)
		#arr[0,0] = first_isupper
		
		return arr
	
	def extract_tsents_features(self, tagged_sentences):
		X = None
		Y = []
		for tagged_sentence in tagged_sentences:
			tsx, tsy = self.extract_tsent_features(tagged_sentence)
			if X == None:
				X = tsx
			else:
				X = vstack([X, tsx], dtype=np.int32)
			Y += tsy
		return (X,Y)
		
	def extract_tsent_features(self, tagged_sentence):
		X = None
		Y = []
		for i, tagged_word in enumerate(tagged_sentence):
			word = tagged_word[0]
			tag = tagged_word[1]
			arr = self.extract_word_features(word, i == 0)
			#exit()
			if X == None:
				X = arr
			else:
				X = vstack([X, arr], dtype=np.int32)
			Y += [tag]
		return (X, Y)
	
	
	
	def tag_sents(self, sents):
		arr = extract_sents_features(word)
		return self.clf.predict(arr)
		
	def evaluate(self, gold):
		X, Y = self.extract_tsents_features(gold)
		Y_pred = self.clf.predict(X)
		print(Y[:10])
		print(Y_pred[:10])
		return accuracy_score(Y, Y_pred)
			
			