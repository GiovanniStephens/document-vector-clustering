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

    def _get_reduced_embeddings(self):
        """Imports embeddings module and reduces the sentence vectors
        to two dimensions."""
        embed = self._get_module('embeddings')
        clustering =  self._get_module('clustering')
        embeddings = embed.pretrained_transformer_embeddings(TEST_UTTERANCES)
        n_dimensions = 2
        reduced = clustering.reduce_dimensions_pca(embeddings, n_dimensions)
        return reduced

    def test_import_embeddings_module(self):
        """Tests that you can import the embeddings module."""
        embed = self._get_module('embeddings')
        self.assertIsNotNone(embed, "Could not import embeddings module.")

    def test_import_clustering_module(self):
        """Tests that you can import the clustering module."""
        clustering = self._get_module('clustering')
        self.assertIsNotNone(clustering, "Could not import clustering module.")

    def test_reduce_dimensions_pca_num_dimensions(self):
        """Tests the size of the vector after dimension reduction using PCA."""
        reduced = self._get_reduced_embeddings()
        self.assertEqual(len(reduced[0]), 2)

    def test_reduce_dimensions_pca_similarity(self):
        """Tests vector similarity by ways of cosine similarity 
        using the reduced vectors.
        The first two test phrases should be more similar than
        the first and third."""
        reduced = self._get_reduced_embeddings()
        sim_1 = self._cosine_similarity(reduced[0], reduced[1])
        sim_2 = self._cosine_similarity(reduced[0], reduced[2])
        self.assertGreater(sim_1, sim_2, "The similarity between 1 and 2 should be more than 1 and 3.")

    def test_reduce_dimensions_umap(self):
        """Tests the size of the vector after dimension reduction using UMAP."""
        embed = self._get_module('embeddings')
        clustering =  self._get_module('clustering')
        embeddings = embed.pretrained_transformer_embeddings(TEST_UTTERANCES*10)
        n_dimensions = 2
        reduced = clustering.reduce_dimensions_umap(embeddings, n_dimensions, n_neighbors=2)
        self.assertEqual(len(reduced[0]), n_dimensions)
    
    def test_get_optimal_n_components(self):
        """Tests the function to find the optimal number of principal
        components to keep."""
        embed = self._get_module('embeddings')
        clustering =  self._get_module('clustering')
        embeddings = embed.pretrained_transformer_embeddings(TEST_UTTERANCES*10)
        n_dimensions = clustering.get_optimal_n_components(embeddings)
        self.assertEqual(n_dimensions, 2)

    def test_kmeans_clustering_diff_cluster(self):
        """Tests that the first phrase is not clustered with the third using kmeans."""
        clustering =  self._get_module('clustering')
        reduced = self._get_reduced_embeddings()
        cluster_labels = clustering.kmeans_clustering(reduced, max_num_clusters = 3)
        self.assertNotEqual(cluster_labels[0], cluster_labels[2])

    def test_hdbscan_clustering_diff_cluster(self):
        """Tests that the first phrase is not clustered with the third using HDBSCAN."""
        clustering =  self._get_module('clustering')
        reduced = self._get_reduced_embeddings()
        cluster_labels = clustering.hdbscan_clustering(reduced, min_cluster_size = 2)
        self.assertEqual(cluster_labels[0], cluster_labels[2])

    def test_kmeans_clustering_same_cluster(self):
        """Tests that the first phrase is clustered with the second using kmeans."""
        clustering =  self._get_module('clustering')
        reduced = self._get_reduced_embeddings()
        cluster_labels = clustering.kmeans_clustering(reduced, max_num_clusters = 3)
        self.assertEqual(cluster_labels[0], cluster_labels[1])

    def test_hdbscan_clustering_same_cluster(self):
        """Tests that the first phrase is clustered with the second using HDBSCAN."""
        clustering =  self._get_module('clustering')
        reduced = self._get_reduced_embeddings()
        cluster_labels = clustering.hdbscan_clustering(reduced, min_cluster_size = 2)
        self.assertEqual(cluster_labels[0], cluster_labels[1])

if __name__ == '__main__':
    unittest.main()