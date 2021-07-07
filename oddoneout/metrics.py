"""
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
"""
from taxonomy import GraphTaxonomy
from taxonomy import lowest_common_ancestor as lowest_ca
#from ozone.taxonomy import WordnetTaxonomy
from wordnet import WordnetTaxonomy
from time import sleep
def flatness(taxonomy, node):
    """
    The ratio of children to total descendents of a category.
    A 'flatter' node will have a higher flatness.

    TODO: consider an alternative based on average branching factor in the
    graph rooted at that node

    """
    non_root_nodes = len(taxonomy.get_descendants(node)) - 1

    non_leaf_nodes = sum([1 for d in taxonomy.get_descendants(node) if taxonomy.is_instance(d) is not True])

    average_branching_factor = non_root_nodes / non_leaf_nodes

    return average_branching_factor


def repetitions(taxonomy, node):
    """
    Revise this so that:

    repetitions 1
    - it counts the number of instances that can be reached via
    more than 1 child. (for fruit example, "entity" == 2) <-- orange and peach

    repetitions 2
    - or: the average number of children through which we can access an instance
    (for fruit example "entity" = (2+2+1+1+1+1) / 6

    """

    # repetitions 1
    total = 0
    instances = taxonomy.get_descendant_instances(node)
    for instance in instances:
        total += len(taxonomy.get_parents(instance))
    return total / len(instances)


    # repetitions 2
    # return sum([1 for instance in taxonomy.get_descendant_instances(node) if len(taxonomy.get_parents(instance)) > 1])

def wup_distance(taxonomy, parent, child):
    curr = parent
    res = 0
    while curr != None:
        
        curr_children = taxonomy.get_children(curr)
        curr_children_as_lemmas = [taxonomy.as_lemma(l) for l in curr_children]

        if curr_children == []:
            return 0
            
        else:
            if child in curr_children_as_lemmas:
                return res + 1
            else:
                for c in curr_children:
                    if child in taxonomy.get_descendants(c):
                        curr = c
                        res += 1
                        break
                return res + 1
                

def wu_palmer_similarity(taxonomy, node1, node2, target):
    """
    Similarity metric from Wu and Palmer (1994).

    Given two nodes node 1 and node 2,
    the Wu Palmer similarity of the two nodes is the depth of the lowest
    common ancestor of the two nodes divided by the sum of the depths of
    the two nodes. This ratio is then multiplied by two.

    """
    target_ancestors = taxonomy.get_ancestor_categories(target)
    common_ancestors = taxonomy.get_ancestor_categories(node1) & taxonomy.get_ancestor_categories(node2)
    common_ancestors = common_ancestors - target_ancestors
    if len(common_ancestors) == 0:
        return 0
    scored_ancestors = [(taxonomy.get_specificity(hyp), hyp)
                        for hyp in common_ancestors]
    sorted_ancestors = sorted(scored_ancestors)
    least_common_ancestor = sorted_ancestors[0][1]
    node1_score = wup_distance(taxonomy, least_common_ancestor, node1)
    node2_score = wup_distance(taxonomy, least_common_ancestor, node2)
    node3_score = wup_distance(taxonomy, target, taxonomy.as_lemma(least_common_ancestor))
    numerator = 2 * node3_score
    denominator = node1_score + node2_score + (2 * node3_score)
    return numerator / denominator

def wu_palmer_similarity_2(taxonomy, node1, node2):
    """
    Similarity metric from Wu and Palmer (1994).

    Given two nodes node 1 and node 2,
    the Wu Palmer similarity of the two nodes is the depth of the lowest
    common ancestor of the two nodes divided by the sum of the depths of
    the two nodes. This ratio is then multiplied by two.

    """
    # Get dicts of hypernym:distance from node to hypernym (AKA index of list)
    node1_ancestor_distances = dict()
    node1_ancestors = (taxonomy.get_ancestors(node1))
    for h in node1_ancestors:
        node1_ancestor_distances[h] = node1_ancestors.index(h)

    node2_ancestor_distances = dict()
    node2_ancestors = (taxonomy.get_ancestors(node2))
    for h in node2_ancestors:
        node2_ancestor_distances[h] = node2_ancestors.index(h)

    # Find the common hypernyms between the two nodes
    common = set(node1_ancestors) & set(node2_ancestors)

    # get sums of distances of common hypernyms, then get the word with minimum sum
    candidates = dict()
    for c in common:
        candidates[c] = node1_ancestor_distances[c] + node2_ancestor_distances[c]
    
    lowest_common_ancestor = min(candidates, key=candidates.get)
    # lowest_common_ancestor = lowest_ca(taxonomy, [node1, node2], 'entity')[1]
    # print("node 1 ancestors: ", node1_ancestors)
    # print("node 2 ancestors: ", node2_ancestors)

    node1_lca_distance = node1_ancestor_distances[lowest_common_ancestor]
    node2_lca_distance = node2_ancestor_distances[lowest_common_ancestor]
    node3_distance = len(taxonomy.get_ancestors(lowest_common_ancestor)) - 1
    numerator = 2 * node3_distance
    denominator = node1_lca_distance + node2_lca_distance + (2 * node3_distance)

    if denominator == 0:
        return 0
    return numerator / denominator

def rosenberg_descendent_similarity(taxonomy, node):
    total = 0
    num_tests = 10
    descendents = taxonomy.get_descendants(node)
    for d in descendents:
        d_sim_total = 0
        for _ in range(num_tests):
            x = taxonomy.random_descendants(node, 1)[0]
            d_sim_total += wu_palmer_similarity_2(taxonomy, d, x)
        d_sim_avg = d_sim_total / num_tests
        total += d_sim_avg
    if len(descendents) == 0:
        return 0
    avg = total / len(descendents)
    return avg