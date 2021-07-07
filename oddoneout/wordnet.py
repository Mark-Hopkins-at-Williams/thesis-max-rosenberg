"""
import nltk
nltk.download('wordnet')
"""

from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import WordNetError
from taxonomy import Taxonomy, Specificity


class WordnetTaxonomy(Taxonomy):

    def __init__(self, root='entity.n.01'):
        self.root = root
        self.specificity = Specificity()
        self.instances = self.get_descendant_instances(self.get_root())
        self.num_insts = len(self.instances)
        sense = wn.synset(root)
        self.categories = {sense.name()}
        for y in sense.hyponyms():
            self.categories.add(y.name())
            for z in get_all_hyponyms_from_sense(y):
                self.categories.add(z.name())

    def is_instance(self, node):
        return node in self.instances and len(wn.synsets(encode_lemma(node))) > 0

    def is_category(self, node):
        return node in self.categories

    def num_instances(self):
        return self.num_insts

    def get_root(self):
        return self.root

    def get_categories(self):
        return self.categories

    def get_instances(self):
        return self.instances

    def get_specificity(self, category):
        return self.specificity(self, category)

    def get_ancestor_categories(self, node):
        if self.is_instance(node):
            synsets = wn.synsets(encode_lemma(node))
        else:
            synsets = [wn.synset(node)]
        result = []
        for synset in synsets:
            result.append(synset.name())
            while len(synset.hypernyms()) > 0:
                synset = synset.hypernyms()[0]
                result.append(synset.name())
        return set([x for x in result if x in self.categories])

    def get_children(self, synset_name):
        sense = wn.synset(synset_name)
        return [syn.name() for syn in sense.hyponyms()]

    def get_descendants(self, synset_name):
        return self.get_descendant_instances(synset_name)

    def get_descendant_instances(self, node):
        result = set()
        sense = wn.synset(node)
        for y in sense.hyponyms():
            for lemma in y.lemmas():
                result.add(decode_lemma(lemma))
            for z in get_all_hyponyms_from_sense(y):
                for lemma in z.lemmas():
                    result.add(decode_lemma(lemma))
        return result

    def as_lemma(self, synset_name):
        return synset_name[:-5]



def decode_lemma(lemma):
    return ' '.join(lemma.name().split("_")).lower()


def encode_lemma(encoded):
    return '_'.join(encoded.split())


def hypernym_chain(synset_name):
    """
    e.g. hypernym_chain('boat.n.01')

    """
    synset = wn.synset(synset_name)
    result = [synset]
    while len(synset.hypernyms()) > 0:
        synset = synset.hypernyms()[0]
        result.append(synset)
    return result


def get_all_hypernyms_from_sense(sense):
    result = set()
    for y in sense.hypernyms():
        result.add(y)
        for z in get_all_hypernyms_from_sense(y):
            result.add(z)
    return result


def get_all_hypernyms(word):
    """
    e.g. get_all_hypernyms('dog')

    """
    result = set()
    for x in wn.synsets(word):
        for y in get_all_hypernyms_from_sense(x):
            result.add(y)
    return result


def get_all_hyponyms_from_sense(sense):
    """
    e.g. get_all_hyponyms_from_sense(wn.synset('metallic_element.n.01'))

    """
    result = set()
    for y in sense.hyponyms():
        result.add(y)
        for z in get_all_hyponyms_from_sense(y):
            result.add(z)
    return result
