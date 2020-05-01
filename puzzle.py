import random
from wordnet import GetRandomSynset, get_all_hyponyms_from_sense
from wordnet import get_all_lemmas_from_sense, normalize_lemma
from nltk.corpus import wordnet as wn
from cuda import FloatTensor, LongTensor, cudaify


class PuzzleGenerator:
    
    def __init__(self):
        pass

    def batch_generate(self, number_of_puzzles = 10):
        return [self.generate() for n in range(number_of_puzzles)]

    def generate(self):
        raise NotImplementedError('cannot call .generate() on abstract class.')

    
class WordnetPuzzleGenerator(PuzzleGenerator):
    
    def __init__(self, root_synset):
        super(WordnetPuzzleGenerator, self).__init__()
        self.root_synset = wn.synset(root_synset)
        self.synset_gen = GetRandomSynset(root_synset)
        self.vocab = self._build_vocab()
        
    def _build_vocab(self):
        words = list(get_all_lemmas_from_sense(self.root_synset))
        word_to_ix = dict([(v, k) for (k,v) in enumerate(words)])
        print("vocab size for word {}: {}".format(self.root_synset.name(), len(word_to_ix)))
        return word_to_ix
    
    def get_vocab(self):
        return self.vocab
    
    def reset_root(self, root_synset):
        self.root_synset = wn.synset(root_synset)
        self.synset_gen = GetRandomSynset(root_synset)        
    
    def generate(self):
        root = self.synset_gen.random_synset_with_specificity(10, 1000)
        while root == self.root_synset:
            root = self.synset_gen.random_synset_with_specificity(10, 1000)
        hyps = get_all_lemmas_from_sense(root) # children?
        puzzle = random.sample(hyps, 4)
        random_hyp = self.synset_gen.random_non_hyponym(root)
        random_word = random.choice(list(get_all_lemmas_from_sense(random_hyp)))
        puzzle.append(random_word)
        (w1, w2, w3, w4, w5) = puzzle#[normalize_lemma(random.choice(s.lemmas()).name()) for s in puzzle]
        result = [(str(w1), 0), (str(w2), 0), (str(w3), 0), 
                  (str(w4), 0), (str(w5), 1)]
        random.shuffle(result)
        xyz = tuple([i for (i,_) in result])
        onehot = [j for (_,j) in result]
        return (xyz, onehot.index(1))

def one_hot(word, vocab):
    vec = [0]*len(vocab)
    vec[vocab[word]] = 1
    return vec

def make_puzzle_vector(puzzle, vocab):
    choices, _ = puzzle
    oneHotVec = []
    for choice in choices:
        oneHotVec += one_hot(str(choice), vocab)
    return cudaify(FloatTensor(oneHotVec).view(1, -1))

def make_puzzle_matrix(puzzles, vocab):
    matrix = []
    for puzzle in puzzles:
        choices, _ = puzzle
        oneHotVec = []
        for choice in choices:
            oneHotVec += one_hot(str(choice), vocab)
        matrix.append(oneHotVec)
    return cudaify(FloatTensor(matrix))

def make_puzzle_target(label):
    return cudaify(LongTensor([label]))

def make_puzzle_targets(labels):
    return cudaify(LongTensor(labels))

