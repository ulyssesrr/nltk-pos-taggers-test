#!/usr/bin/env python3
# vim:fileencoding=utf-8

import importlib
import os

from mac_morpho import OriginalMacMorphoCorpusReader

from u_tagger import UTagger

import nltk.corpus

import numpy as np

from sklearn.cross_validation import KFold

corpus = dir(nltk.corpus)
corpus = dir(nltk.corpus)

#corpus = os.listdir( nltk.data.find("corpora") )
#corpus = ['mac_morpho']
corpus = [OriginalMacMorphoCorpusReader()]

for c in corpus:
	#try:
	print("Loading corpora: %s" % (c))
	#mod = __import__('nltk.corpus', fromlist=[c])
	#clazz = getattr(mod, c)
	
	tsents = np.array(c.tagged_sents())
	twords = []
	#print(tsents)
	#twords = np.array(clazz.tagged_words())
	#tsents = np.array(clazz.tagged_sents())
	len_tsents = len(tsents)
	print("%s: Words: %d | Sentences: %d" % (c, len(twords), len_tsents))
	kf = KFold(len_tsents, n_folds=5)
	
	#tag_fd = nltk.FreqDist(tag for (word, tag) in twords)
	t0 = nltk.DefaultTagger('ZZZ')
	for k, (train_index, test_index) in enumerate(kf):
		print("+ WK: %d" % (k))
		tsents_train = tsents[train_index].tolist()
		tsents_test = tsents[test_index].tolist()
		
		tagger0 = UTagger(tsents_train)
		tagger0._taggers = [t0]
		acc = tagger0.evaluate(tsents_test)
		print("|-UTagger: %.04f" % (acc))
		
		tagger1 = nltk.UnigramTagger(tsents_train, backoff=tagger0)
		acc = tagger1.evaluate(tsents_test)
		print("|-Majority+Unigram: %.04f" % (acc))
		
		tagger2 = nltk.BigramTagger(tsents_train, backoff=tagger1)
		acc = tagger2.evaluate(tsents_test)
		print("|-Majority+Unigram+Bigram: %.04f" % (acc))
		
		tagger3 = nltk.TrigramTagger(tsents_train, backoff=tagger2)
		acc = tagger3.evaluate(tsents_test)
		print("|-Majority+Unigram+Bigram+Trigram: %.04f" % (acc))
		exit()
			
#	except Exception as e:
#		print("Erro: %s: %s" % (c,e))
#		None