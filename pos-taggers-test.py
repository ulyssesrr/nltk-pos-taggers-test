#!/usr/bin/env python

import importlib
import nltk.corpus
import os

corpus = dir(nltk.corpus)
corpus = dir(nltk.corpus)

#corpus = os.listdir( nltk.data.find("corpora") )
corpus = ['mac_morpho']

for c in corpus:
	try:
		mod = __import__('nltk.corpus', fromlist=[c])
		clazz = getattr(mod, c)
		print("%s: %d" % (c, len(clazz.tagged_words())))
	except Exception as e:
		#print("Erro: %s: %s" % (c,e))
		None
