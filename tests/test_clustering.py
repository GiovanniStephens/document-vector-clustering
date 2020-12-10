import unittest

TEST_UTTERANCES = (
    'This is a test sentence.'
    , 'This is a similar test phrase.'
    , 'Nothing to do with the others.'
)

class test_embeddings(unittest.TestCase):

    def _get_module(self, module_name):
        import importlib
        return importlib.import_module(module_name)

    def _cosine_similarity(self, v1,v2):
        "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
        sumxx, sumxy, sumyy = 0, 0, 0
        for i in range(len(v1)):
            x = v1[i]; y = v2[i]
            sumxx += x*x
            sumyy += y*y
            sumxy += x*y
        return sumxy/(sumxx*sumyy)**0.5

    def test_import_embeddings_module(self):
        embed = self._get_module('embeddings')
        self.assertIsNotNone(embed)

    def test_reduce_dimensions_pca_num_dimensions(self):
        embed = self._get_module('embeddings')
        clustering =  self._get_module('clustering')
        embeddings = embed.pretrained_transformer_embeddings(TEST_UTTERANCES)
        n_dimensions = 3
        reduced = clustering.reduce_dimensions_pca(embeddings, n_dimensions)
        self.assertEqual(len(reduced[0]), n_dimensions)

    def test_reduce_dimensions_pca_similarity(self):
        embed = self._get_module('embeddings')
        clustering =  self._get_module('clustering')
        embeddings = embed.pretrained_transformer_embeddings(TEST_UTTERANCES)
        n_dimensions = 3
        reduced = clustering.reduce_dimensions_pca(embeddings, n_dimensions)
        sim_1 = self._cosine_similarity(reduced[0], reduced[1])
        sim_2 = self._cosine_similarity(reduced[0], reduced[2])
        self.assertGreater(sim_1, sim_2)

    def test_reduce_dimensions_umap(self):
        embed = self._get_module('embeddings')
        clustering =  self._get_module('clustering')
        embeddings = embed.pretrained_transformer_embeddings(TEST_UTTERANCES*10)
        n_dimensions = 3
        reduced = clustering.reduce_dimensions_umap(embeddings, n_dimensions, n_neighbors=2)
        self.assertEqual(len(reduced[0]), n_dimensions)
    
if __name__ == '__main__':
    unittest.main()