import unittest
from ozone.taxonomy import AnimalTaxonomy, InstanceNode, CategoryNode, TaxonomyNode, Taxonomy

class TestBasicTaxonomy(unittest.TestCase):
    
    def setUp(self):
        animal = CategoryNode("animal")
        bird = CategoryNode("bird")
        mammal = CategoryNode("mammal")
        reptile = CategoryNode("reptile")

        finch = InstanceNode("finch")
        swallow = InstanceNode("swallow")
        dog = CategoryNode("dog")
        cat = InstanceNode("cat")
        monkey = InstanceNode("monkey")
        giraffe = InstanceNode("giraffe")
        iguana = InstanceNode("iguana")

        bulldog = InstanceNode("bulldog")
        poodle = InstanceNode("poodle")

        vertices = [animal, bird, mammal, reptile, finch, swallow, dog,
                        cat, monkey, giraffe, iguana, bulldog, poodle]

        edges = [(animal, bird), (animal, mammal), (animal, reptile),
                (mammal, dog), (mammal, cat), (mammal, monkey),
                (mammal, giraffe), (bird, finch), (bird, swallow),
                (reptile, iguana), (dog, bulldog), (dog, poodle)]

        self.taxonomy = AnimalTaxonomy()

    def test_get_vocab(self):
        expected = {'animal': 0, 'bird': 1, 'mammal': 2, 'reptile': 3,
                    'finch': 4, 'swallow': 5, 'dog': 6, 'cat': 7,
                    'monkey': 8, 'giraffe': 9, 'iguana': 10, 'bulldog': 11,
                    'poodle': 12}
        assert self.taxonomy.get_vocab() == expected

    def test_get_root_synset(self):
        expected = "animal"
        assert self.taxonomy.get_root_node() == expected

    def test_random_node(self):
        assert self.taxonomy.random_node(12,14) == 'animal'

    def test_random_descendants(self):
        expected = set(['bulldog', 'poodle'])
        result = set(self.taxonomy.random_descendants('dog', 2))
        assert result == expected
        expected1 = set(['finch'])
        expected2 = set(['swallow'])
        result = set(self.taxonomy.random_descendants('bird', 1))
        assert result == expected1 or result == expected2

    def test_random_non_descendant(self):
        non_birds = {'dog', 'monkey', 'reptile', 'cat', 
                     'animal', 'mammal', 'iguana', 'poodle',
                     'bulldog', 'giraffe'}
        for _ in range(10):
            non_hyponym = self.taxonomy.random_non_descendant('bird')
            assert non_hyponym in non_birds

    def test_flatness(self):
        assert self.taxonomy.flatness("dog") == 1
        assert self.taxonomy.flatness("mammal") == 2/3
        assert self.taxonomy.flatness("animal") == 1/4

    def test_repetitions(self):
        assert self.taxonomy.repetitions('animal') == 1
        assert self.taxonomy.repetitions('mammal') == 1
        assert self.taxonomy.repetitions('dog') == 1
        assert self.taxonomy.repetitions('bulldog') == 1

        
if __name__ == "__main__":
    unittest.main()
