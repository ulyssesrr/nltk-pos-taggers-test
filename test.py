from u_tagger import UTagger
from mac_morpho import MacMorphoCorpusReader2

mr = MacMorphoCorpusReader2()
print(mr.words.shape)
print(mr.words[:10])
print(mr.tags.shape)
print(mr.tags[:10])
print(mr.sentences.shape)
print(mr.sentences[:10])
print(mr.paragraphs.shape)
print(mr.paragraphs[:10])

#ut = UTagger(tsents_train)
#print(ut.evaluate(tsents_test))