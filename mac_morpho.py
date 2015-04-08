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