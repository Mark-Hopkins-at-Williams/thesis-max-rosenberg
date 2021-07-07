import unittest
from taxonomy import GraphTaxonomy, lowest_common_ancestor
from metrics import flatness, wu_palmer_similarity, repetitions, rosenberg_descendent_similarity
from ozone.taxonomy import WordnetTaxonomy

class TestMetrics(unittest.TestCase):

    def setUp(self):
        self.taxonomy = GraphTaxonomy(
            'entity',
            {'apple': ['fruit'],
             'lemon': ['citrus'],
             'orange': ['citrus', 'color'],
             'peach': ['fruit', 'color'],
             'red': ['color'],
             'yellow': ['color'],
             'citrus': ['fruit'],
             'fruit': ['entity'],
             'color': ['entity'],
             'entity': []}
        )
        self.wnt = WordnetTaxonomy("entity.n.01")

    def test_num_instances(self):
        assert self.taxonomy.num_instances() == 6

    def test_flatness(self):
        assert flatness(self.taxonomy, 'fruit') == 5 / 2
        assert flatness(self.taxonomy, 'color') == 4 / 1
        assert flatness(self.taxonomy, 'entity') == 9 / 4

    def test_repetitions(self):
        assert repetitions(self.taxonomy, 'fruit') == 3 / 2
        assert repetitions(self.taxonomy, 'color') == 3 / 2
        assert repetitions(self.taxonomy, 'entity') == 8 / 6

    def test_wu_palmer(self):
        # TODO: check to make sure these numbers are correct for the
        # example fruit taxonomy.
        print(wu_palmer_similarity(self.taxonomy, "orange", "lemon", "entity"))
        print(wu_palmer_similarity(self.taxonomy, "orange", "red", "entity"))

    def test_rosenberg_descendant_sim(self):
        rds_geological_formation = rosenberg_descendent_similarity(self.wnt, "geological_formation.n.01")
        assert rds_geological_formation < 0.57 and rds_geological_formation > 0.55
        rds_dog = rosenberg_descendent_similarity(self.wnt, "dog.n.01")
        assert rds_dog < 0.85 and rds_dog > 0.83





