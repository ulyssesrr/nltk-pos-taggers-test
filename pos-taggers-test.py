#!/usr/bin/env python

import importlib
import os

import nltk.corpus

import numpy as np

from sklearn.cross_validation import KFold

corpus = dir(nltk.corpus)
corpus = dir(nltk.corpus)

#corpus = os.listdir( nltk.data.find("corpora") )
corpus = ['mac_morpho']

for c in corpus:
	try:
		print("Loading corpora: %s" % (c))
		mod = __import__('nltk.corpus', fromlist=[c])
		clazz = getattr(mod, c)
		twords = np.array(clazz.tagged_words())
		tsents = np.array(clazz.tagged_sents())
		len_tsents = len(tsents)
		print("%s: Words: %d | Sentences: %d" % (c, len(twords), len_tsents))
		kf = KFold(len_tsents, n_folds=10)
		
		tag_fd = nltk.FreqDist(tag for (word, tag) in twords)
		tagger0 = nltk.DefaultTagger('N')
		for k, (train_index, test_index) in enumerate(kf):
			print("+ K: %d" % (k))
			tsents_train = tsents[train_index]
			tsents_test = tsents[test_index]
			
			acc = nltk.tag.accuracy(tagger0, tsents_test)
			print("|-Majority: %.04f" % (acc))
			
			tagger1 = nltk.UnigramTagger(tsents_train, backoff=tagger0)
			acc = nltk.tag.accuracy(tagger1, tsents_test)
			print("|-Majority+Unigram: %.04f" % (acc))
			
			tagger2 = nltk.BigramTagger(tsents_train, backoff=tagger1)
			acc = nltk.tag.accuracy(tagger2, tsents_test)
			print("|-Majority+Unigram+Bigram: %.04f" % (acc))
			
			tagger3 = nltk.TrigramTagger(tsents_train, backoff=tagger2)
			acc = nltk.tag.accuracy(tagger3, tsents_test)
			print("|-Majority+Unigram+Bigram+Trigram: %.04f" % (acc))
			exit()
			
	except Exception as e:
		print("Erro: %s: %s" % (c,e))
		None