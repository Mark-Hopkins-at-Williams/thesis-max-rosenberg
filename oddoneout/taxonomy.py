"""
import nltk
nltk.download('wordnet')
"""

from nltk.corpus import wordnet as wn
from collections import defaultdict


class Taxonomy:

    def is_instance(self, node):
        raise NotImplementedError('Cannot call this method on abstract class.')

    def is_category(self, node):
        raise NotImplementedError('Cannot call this method on abstract class.')

    def num_instances(self):
        raise NotImplementedError('Cannot call this method on abstract class.')

    def get_root(self):
        raise NotImplementedError('Cannot call this method on abstract class.')

    def get_categories(self):
        return NotImplementedError('Cannot call this method on abstract class.')

    def get_instances(self):
        return NotImplementedError('Cannot call this method on abstract class.')

    def get_ancestor_categories(self, node):
        raise NotImplementedError('Cannot call this method on abstract class.')

    def get_descendant_instances(self, node):
        raise NotImplementedError('Cannot call this method on abstract class.')

    def get_descendants(self, node):
        raise NotImplementedError('Cannot call this method on abstract class.')


class Specificity:
    def __init__(self):
        self.cache = dict()

    def __call__(self, taxonomy, category):
        if category not in self.cache:
            spec = len(taxonomy.get_descendant_instances(category))
            self.cache[category] = spec
        return self.cache[category]


def lowest_common_ancestor(taxonomy, words, target):
    target_ancestors = taxonomy.get_ancestor_categories(target)
    common_ancestors = taxonomy.get_ancestor_categories(words[0])
    for word in words[1:]:
        word_ancestors = taxonomy.get_ancestor_categories(word)
        common_ancestors = common_ancestors & word_ancestors
    common_ancestors = common_ancestors - target_ancestors
    if len(common_ancestors) == 0:
        return taxonomy.num_instances(), taxonomy.get_root()
    scored_ancestors = [(taxonomy.get_specificity(hyp), hyp)
                        for hyp in common_ancestors]
    sorted_ancestors = sorted(scored_ancestors)
    return sorted_ancestors[0]


class GraphTaxonomy:

    def __init__(self, root, parents):
        self.specificity = Specificity()
        self.root = root
        self.parents = parents
        self.children = defaultdict(list)
        for node in self.parents:
            for parent in self.parents[node]:
                self.children[parent].append(node)
        self.children = dict(self.children)

    def is_instance(self, node):
        return node in self.parents and node not in self.children

    def is_category(self, node):
        return node in self.children

    def num_instances(self):
        return len(self.parents) - len(self.children)

    def get_root(self):
        return self.root

    def get_categories(self):
        return self.children.keys()

    def get_instances(self):
        return self.parents.keys() - self.children.keys()

    def get_children(self, node):
        if node not in self.children:
            return []
        else:
            return self.children[node]

    def get_parents(self, node):
        if node not in self.parents:
            return []
        else:
            return self.parents[node]

    def get_specificity(self, category):
        return self.specificity(self, category)

    def get_ancestor_categories(self, node):
        result = set()
        if self.is_category(node):
            result.add(node)
        for parent in self.get_parents(node):
            result |= set(self.get_ancestor_categories(parent))
        return result

    def get_descendant_instances(self, node):
        """TODO: optimize!"""
        if self.is_instance(node):
            return {node}
        else:
            result = set()
            for child in self.get_children(node):
                result |= self.get_descendant_instances(child)
            return result

    def get_descendants(self, node):
        """TODO: optimize!"""
        if self.is_instance(node):
            return {node}
        else:
            result = {node}
            for child in self.get_children(node):
                result |= self.get_descendants(child)
            return result

    def as_lemma(self, node):
        return node





