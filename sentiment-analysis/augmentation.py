# 1. amaç => Veriyi artırmak
# 2. amaç => Modelin aynı veriyi farklı varyasyonlarla öğrenmesi.

# Döngü ile tüm veriyi aynı fonk. vermek.

import nlpaug.augmenter.word as naw
import random

random.seed(35)

text = "The weather is nice today and we should go for a walk."

# Random Synonym Augmenter
synonm_aug = naw.SynonymAug(aug_src="wordnet")
text_synonym = synonm_aug.augment(text)
#
print(text_synonym)

# Random Swap
swap_aug = naw.RandomWordAug(action="swap")
text_swap = swap_aug.augment(text)
print(text_swap)
#

# Random Deletion
deletion_aug = naw.RandomWordAug(action="delete")
text_deletion = deletion_aug.augment(text)
print(text_deletion)
#

#
#bert_insert = naw.ContextualWordEmbsAug(model_path="bert-base-uncased", action="insert")
#text_bert_insert = bert_insert.augment(text)
#print(text_bert_insert)
#


