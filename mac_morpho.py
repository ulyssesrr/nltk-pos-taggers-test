# vim:fileencoding=utf-8

class OriginalMacMorphoCorpusReader:

	def __init__(self):
		self.train_tagged_sents = self.read('corpora/mac_morpho/macmorpho-train.txt')
		self.test_tagged_sents = self.read('corpora/mac_morpho/macmorpho-test.txt')
		
	
	def read(self, fname):
		with open(fname, encoding="utf-8") as f:
			content = f.readlines()
		#print(content[:10])
		#print(list(map(lambda line : list(map(lambda tagged_word : tagged_word.split('_'), line.split())), content[:10])))
		return list(map(lambda line : list(map(lambda tagged_word : tuple(tagged_word.split('_')), line.split())), content))
		
	def tagged_sents(self):
		return self.train_tagged_sents + self.test_tagged_sents
	

import itertools
import numpy as np

class MacMorphoCorpusReader2:

	def __init__(self):
		train_words, train_tags, train_sentences = self.read('corpora/mac_morpho/macmorpho-train.txt')
		test_words, test_tags, test_sentences = self.read('corpora/mac_morpho/macmorpho-test.txt')
		self.words = np.hstack((train_words, test_words))
		self.tags = np.hstack((train_tags, test_tags))
		test_sentences = np.array(test_sentences)
		test_sentences = test_sentences + train_sentences[-1][1]
		self.sentences = np.vstack((train_sentences, test_sentences))
		# no paragraph information each sentence is an paragraph
		len_sentences = len(self.sentences)
		self.paragraphs = np.column_stack((np.arange(len_sentences), np.arange(start=1, stop=1+len_sentences)))
	
	def read(self, fname):
		with open(fname, encoding="utf-8") as f:
			content = f.readlines()
			data = [line.replace('_', ' ').split() for line in content]
			words = [line[::2] for line in data]
			tags = [line[1::2] for line in data]
			i = 0
			sentences = []
			for sentence in words:
				len_sentence = len(sentence)
				sentences += [[i, i + len_sentence]]
				i += len_sentence
			words = list(itertools.chain.from_iterable(words))
			tags = list(itertools.chain.from_iterable(tags))
			return (words, tags, sentences)
		
	def tagged_sents(self):
		return self.train_tagged_sents + self.test_tagged_sents