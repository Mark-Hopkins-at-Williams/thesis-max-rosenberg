import random
import queue
from ozone.animals import AnimalNet
from ozone.wordnet import GetRandomSynset
from ozone.wordnet import get_all_lemmas_from_sense, hypernym_chain, Specificity
from ozone.puzzle import PuzzleGenerator
from nltk.corpus import wordnet as wn

class TaxonomyNode:
    def __init__(self, name):
        self.name = name

class CategoryNode(TaxonomyNode):
    def __init__(self, name):
        super(CategoryNode, self).__init__(name)

class InstanceNode(TaxonomyNode):
    def __init__(self, name):
        super(InstanceNode, self).__init__(name)

class TaxonomyGraph:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges
        vertex_to_ix = dict([(v.name, k) for (k,v) in enumerate(self.nodes)])
        self.vocab = vertex_to_ix
        self.name_to_vertex = dict([(v.name, v) for v in (self.nodes)])

class Taxonomy:

    def get_root_node(self):
        raise NotImplementedError("Cannot call this on an abstract class.")
    
    def get_vocab(self):
        raise NotImplementedError("Cannot call this on an abstract class.")

    def get_descendants(self, node):
        raise NotImplementedError("Cannot call this on an abstract class.")

    def get_children(self, node):
        raise NotImplementedError("Cannot call this on an abstract class.")

    def get_ancestors(self, node):
        raise NotImplementedError("Cannot call this on an abstract class.")

    def get_specificity(self, node):
        return len(self.get_descendants(node)) + 1

    def random_node(self, specificity_lb, specificity_ub):
        shuffled = [x for x in self.get_descendants(self.get_root_node())]
        shuffled.append(self.get_root_node())
        random.shuffle(shuffled)
        for element in shuffled:
            spec = self.get_specificity(element)
            if spec <= specificity_ub and spec >= specificity_lb:
                return element
        raise Exception("Couldn't find a node with specificity within the bounds")

    def random_non_descendant(self, node):
        counter = 0
        while (counter < 1000):
            check_node = (self.random_node(0, 10))
            hyps = self.get_descendants(node)
            if check_node == node or check_node in hyps:
                counter += 1
            else:
                return check_node

    def random_descendants(self, node, k):
        # node = wn.synset(node)
        hyps = self.get_descendants(node) # children?
        if len(hyps) < k:
            return None
        return random.sample(hyps, k)

    def flatness(self, node):
        """
        The ratio of children to total descendants of a node.
        A 'flatter' node will have a higher flatness.
        """
        num_total_hyps = len(self.get_descendants(node))
        if num_total_hyps == 0:
            return 0
        return len(self.get_children(node)) / num_total_hyps

    def repetitions(self, node):
        """
        Repititions  is the number of times a given node appears in the taxonomy.
        """
        if node == self.get_root_node():
            return self.get_descendants(self.get_root_node()).count(node) + 1
        else:
            return self.get_descendants(self.get_root_node()).count(node)

    def wu_palmer_similarity(self, node1, node2):
        """
        Similarity metric from Wu and Palmer (1994).

        Given two nodes node 1 and node 2,
        the Wu Palmer similarity of the two nodes is the depth of the lowest common ancestor
        of the two nodes divided by the sum of the depths of the two nodes. This ratio
        is then multiplied by two.

        """
        # Get dicts of hypernym:distance from node to hypernym (AKA index of list)
        node1_ancestor_distances = dict()
        node1_ancestors = self.get_ancestors(node1)
        for h in node1_ancestors:
            node1_ancestor_distances[h] = node1_ancestors.index(h) 

        node2_ancestor_distances = dict()
        node2_ancestors = self.get_ancestors(node2)
        for h in node2_ancestors:
            node2_ancestor_distances[h] = node2_ancestors.index(h) 

        # Find the common hypernyms between the two nodes
        common = set(node1_ancestors) & set(node2_ancestors)

        #get sums of distances of common hypernyms, then get the word with minimum sum
        candidates = dict()
        for c in common:
            candidates[c] = node1_ancestor_distances[c] + node2_ancestor_distances[c]
            
        lowest_common_ancestor = min(candidates, key=candidates.get)

        node1_lca_distance = node1_ancestor_distances[lowest_common_ancestor]
        node2_lca_distance = node2_ancestor_distances[lowest_common_ancestor]
        node3_distance = len(self.get_ancestors(lowest_common_ancestor)) - 1
        numerator = 2 * node3_distance
        denominator = node1_lca_distance + node2_lca_distance + (2 * node3_distance)
        print(node1_lca_distance, node2_lca_distance, node3_distance)

        if denominator == 0:
            return 0

        return numerator / denominator

    def rosenberg_descendant_similarity(self, node):
        total = 0
        num_tests = 10
        descendants = self.get_descendants(node)
        for d in descendants:
            d_sim_total = 0
            for _ in range(num_tests):
                x = self.random_descendants(node, 1)[0]
                d_sim_total += self.wu_palmer_similarity(d,x)
            d_sim_avg = d_sim_total / num_tests 
            total += d_sim_avg
        if len(descendants) == 0:
            return 0
        avg = total / len(descendants)
        return avg

    def wu_palmer_similarity_based_5thing_ooo_solver(self, nodes):
        assert len(nodes) == 5
        nodes = [node.replace(" ", "_") for node in nodes]
        node1 = wn.synset(nodes[0] + ".n.01")
        node2 = wn.synset(nodes[1] + ".n.01")
        node3 = wn.synset(nodes[2] + ".n.01")
        node4 = wn.synset(nodes[3] + ".n.01")
        node5 = wn.synset(nodes[4] + ".n.01")
        similarities = dict()
        n1n2 = self.wu_palmer_similarity(node1, node2)
        n1n3 = self.wu_palmer_similarity(node1, node3)
        n1n4 = self.wu_palmer_similarity(node1, node4)
        n1n5 = self.wu_palmer_similarity(node1, node5)
        node1_sim = n1n2+n1n3+n1n4+n1n5
        similarities[node1] = node1_sim

        n2n1 = self.wu_palmer_similarity(node2, node1)
        n2n3 = self.wu_palmer_similarity(node2, node3)
        n2n4 = self.wu_palmer_similarity(node2, node4)
        n2n5 = self.wu_palmer_similarity(node2, node5)
        node2_sim = n2n1+n2n3+n2n4+n2n5
        similarities[node2] = node2_sim

        n3n1 = self.wu_palmer_similarity(node3, node1)
        n3n2 = self.wu_palmer_similarity(node3, node2)
        n3n4 = self.wu_palmer_similarity(node3, node4)
        n3n5 = self.wu_palmer_similarity(node3, node5)
        node3_sim = n3n1+n3n2+n3n4+n3n5
        similarities[node3] = node3_sim

        n4n1 = self.wu_palmer_similarity(node4, node1)
        n4n2 = self.wu_palmer_similarity(node4, node2)
        n4n3 = self.wu_palmer_similarity(node4, node3)
        n4n5 = self.wu_palmer_similarity(node4, node5)
        node4_sim = n4n1+n4n2+n4n3+n4n5
        similarities[node4] = node4_sim

        n5n1 = self.wu_palmer_similarity(node5, node1)
        n5n2 = self.wu_palmer_similarity(node5, node2)
        n5n3 = self.wu_palmer_similarity(node5, node3)
        n5n4 = self.wu_palmer_similarity(node5, node4)
        node5_sim = n5n1+n5n2+n5n3+n5n4
        similarities[node5] = node5_sim
        res = min(similarities, key=similarities.get)
        return res.name()[:-5].replace("_"," ")


class AnimalTaxonomy(Taxonomy):
    '''Basic Taxonomy of Animals'''
    
    def __init__(self):
        self.an = AnimalNet()
        self.root_synset = self.an.get_animal("animal")
        self.vocab = self.an.graph.vocab

    def get_root_node(self):
        return self.root_synset.name

    def get_vocab(self):
        return self.vocab

    def get_descendants(self, node):
        node = self.an.get_animal(node)
        return [d.name for d in self.an.graph.descendants(node)]
    
    def get_children(self, node):
        node = self.an.get_animal(node)
        return [child.name for child in self.an.graph.children(node)]

    def get_ancestors(self, node):
        node_obj = self.an.get_animal(node)
        result = []
        for y in self.an.graph.ancestors(node_obj):
            result.append(y.get_name())
        return result

class WordnetTaxonomy(Taxonomy):
    
    def __init__(self, root_synset_name):
        # super().__init__()
        self.root_node = wn.synset(root_synset_name)
        self.synset_gen = GetRandomSynset(root_synset_name)
        self.vocab = self._build_vocab()

    def _build_vocab(self):
        words = sorted(list(get_all_lemmas_from_sense(self.root_node)))
        word_to_ix = dict([(v, k) for (k,v) in enumerate(words)])
        # print("vocab size: {}".format(len(word_to_ix)))
        return word_to_ix

    def get_root_node(self):
        return self.root_node.name()
        
    def get_vocab(self):
        return self.vocab

    def get_descendants(self, node):
        node = wn.synset(node)
        result = set()
        for y in node.hyponyms():
            result.add(y.name())
            for z in self.get_descendants(y.name()):
                result.add(z)
        return result

    def get_children(self, node):
        node_obj = wn.synset(node)
        return [hypo.name() for hypo in node_obj.hyponyms()]

    def get_ancestors(self, node):
        return [hyper.name() for hyper in hypernym_chain(node)]

class TaxonomyPuzzleGenerator(PuzzleGenerator):
    
    def __init__(self, taxonomy, num_choices):
        super().__init__()
        self.taxonomy = taxonomy
        self.specificity_lb = 3
        self.specificity_ub = 10
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
        puzzle = self.taxonomy.random_descendants(root, self.num_choices() - 1)
        while not puzzle:
            root = self.taxonomy.random_node(self.specificity_lb, 
                                         self.specificity_ub)
            puzzle = self.taxonomy.random_descendants(root, self.num_choices() - 1)
        
        random_word = self.taxonomy.random_non_descendant(root)
        print('random word: ', random_word)
        puzzle.append(random_word)
        result = [(str(choice), 0) for choice in puzzle[:-1]] + [(str(puzzle[-1]), 1)]
        random.shuffle(result)
        xyz = tuple([i for (i, _) in result])
        onehot = [j for (_, j) in result]    
        return (xyz, onehot.index(1))

if __name__ == "__main__":
    ant = AnimalTaxonomy()
    wnt = WordnetTaxonomy("apple.n.01")
    print(wnt.get_specificity('eating_apple.n.01'))