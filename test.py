from u_tagger import UTagger
from nltk.corpus import mac_morpho

tsents_train = mac_morpho.tagged_sents()[:100]
tsents_test = mac_morpho.tagged_sents()[100:200]

ut = UTagger(tsents_train)
print(ut.evaluate(tsents_test))