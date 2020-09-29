import random
from ozone.animals import AnimalWord, AnimalNet
from ozone.wordnet import GetRandomSynset
from ozone.wordnet import get_all_lemmas_from_sense
from ozone.puzzle import PuzzleGenerator
from nltk.corpus import wordnet as wn

class Taxonomy:
    
    def get_vocab(self):
        raise NotImplementedError("Cannot call this on an abstract class.")

    def get_root_synset(self):
        raise NotImplementedError("Cannot call this on an abstract class.")

    def get_all_hyponyms(self, node):
        raise NotImplementedError("Cannot call this on an abstract class.")

    def get_hyponyms(self, node):
        raise NotImplementedError("Cannot call this on an abstract class.")

    def random_node(self, specificity_lb, specificity_ub):
        raise NotImplementedError("Cannot call this on an abstract class.")

    def random_hyponyms(self, node, k):
        raise NotImplementedError("Cannot call this on an abstract class.")
        
    def random_non_hyponym(self, node):
        raise NotImplementedError("Cannot call this on an abstract class.")

    def flatness(self, node):
        num_total_hyps = len(self.get_all_hyponyms(node))
        if num_total_hyps == 0:
            return 0
        return len(self.get_hyponyms(node)) / num_total_hyps

    def repetitions(self, node): 
        if node == self.get_root_synset():
            return self.get_all_hyponyms(self.get_root_synset()).count(node) + 1
        else:
            return self.get_all_hyponyms(self.get_root_synset()).count(node)


class AnimalTaxonomy(Taxonomy):
    '''Basic Taxonomy of Animals'''
    
    def __init__(self):
        self.an = AnimalNet()
        self.root_synset = self.an.get_animal("animal")
        self.vocab = self.an.vocab

    def get_root_synset(self):
        return self.root_synset.get_name()

    def get_vocab(self):
        return self.vocab

    def get_specificity(self, node):
        return len(self.get_all_hyponyms(node)) + 1

    def get_hyponyms(self, node):
        node = self.an.get_animal(node)
        return node.hyponyms

    def get_all_hyponyms(self, node):
        result = []
        node = self.an.get_animal(node)
        if node != None:
            for y in self.get_hyponyms(node):
                result.append(y)
                for z in self.get_all_hyponyms(y):
                    result.append(z)
        return result

    def random_node(self,specificity_lb, specificity_ub):
        shuffled_animals = self.an.animal_list
        random.shuffle(shuffled_animals)
        for animal in shuffled_animals:
            spec = self.get_specificity(animal)
            if spec < specificity_ub and spec > specificity_lb:
                return animal.name
        raise Exception("Couldn't find a node with specificity within the bounds")

    def random_hyponyms(self, node, k):
        hyps = [hypo.name for hypo in self.get_all_hyponyms(node)]
        return random.sample(hyps, k)

    def random_non_hyponym(self, node):
        counter = 0
        node = self.an.get_animal(node)
        while (counter < 1000):
            check_node = self.an.get_animal(self.random_node(0, 10))
            hyps = self.get_all_hyponyms(node)
            if check_node.name == node.name or check_node in hyps:
                counter += 1
            else:
                return check_node.name

    def repetitions(self, node):
        node = self.an.get_animal(node)
        if node.name == self.get_root_synset():
            return self.get_all_hyponyms(self.get_root_synset()).count(node) + 1

        return self.get_all_hyponyms(self.get_root_synset()).count(node)

class WordnetTaxonomy(Taxonomy):
    
    def __init__(self, root_synset_name):
        super().__init__()
        self.root_synset = wn.synset(root_synset_name)
        self.synset_gen = GetRandomSynset(root_synset_name)
        self.vocab = self._build_vocab()

    def get_root_synset(self):
        return self.root_synset.name()
        
    def get_vocab(self):
        return self.vocab

    def get_all_hyponyms(self, node):
        result = set()
        for y in node.hyponyms():
            result.add(y)
            for z in self.get_all_hyponyms(y):
                result.add(z)
        return result

    def get_hyponyms(self, synset_name):
        sense = wn.synset(synset_name)
        return sense.hyponyms()

    def _build_vocab(self):
        words = sorted(list(get_all_lemmas_from_sense(self.root_synset)))
        word_to_ix = dict([(v, k) for (k,v) in enumerate(words)])
        # print("vocab size: {}".format(len(word_to_ix)))
        return word_to_ix

    def random_node(self, specificity_lb, specificity_ub):
        root = self.synset_gen.random_synset_with_specificity(specificity_lb, 
                                                              specificity_ub)
        if root is None:
            return None
        while root == self.root_synset:
            root = self.synset_gen.random_synset_with_specificity(specificity_lb,
                                                                  specificity_ub)
        return root.name()
    
    def random_hyponyms(self, node, k):
        node = wn.synset(node)
        hyps = get_all_lemmas_from_sense(node) # children?
        return random.sample(hyps, k)
    
    def random_non_hyponym(self, node):
        node = wn.synset(node)
        hyps = get_all_lemmas_from_sense(node) # children?
        counter = 0
        random_hyp_lemmas = []
        while len(random_hyp_lemmas) == 0:
            if counter > 10000:
                raise Exception('Too difficult to get a non-hyponym for {}; giving up'.format(node.name()))
            random_hyp = self.synset_gen.random_non_hyponym(node.name())
            random_hyp_lemmas = set(get_all_lemmas_from_sense(random_hyp))
            random_hyp_lemmas -= hyps
            counter += 1
        random_word = random.choice(list(random_hyp_lemmas))
        return random_word

class TaxonomyPuzzleGenerator(PuzzleGenerator):
    
    def __init__(self, taxonomy, num_choices):
        super().__init__()
        self.taxonomy = taxonomy
        self.specificity_lb = 10
        self.specificity_ub = 5000
        self.n_choices = num_choices
   
    def num_choices(self):
        return self.n_choices

    def max_tokens_per_choice(self):
        return 1
    
    def get_vocab(self):
        return self.taxonomy.get_vocab()
        
    def generate(self):
        root = self.taxonomy.random_node(self.specificity_lb, 
                                         self.specificity_ub)
        puzzle = self.taxonomy.random_hyponyms(root, self.num_choices() - 1)
        random_word = self.taxonomy.random_non_hyponym(root)
        puzzle.append(random_word)
        result = [(str(choice), 0) for choice in puzzle[:-1]] + [(str(puzzle[-1]), 1)]
        random.shuffle(result)
        xyz = tuple([i for (i, _) in result])
        onehot = [j for (_, j) in result]    
        return (xyz, onehot.index(1))
    
