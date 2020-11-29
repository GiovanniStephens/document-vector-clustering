from sklearn.decomposition import PCA
import umap.umap_ as umap
import hdbscan
from sklearn.cluster import KMeans #For Kmeans clustering
from sklearn import cluster
from scipy.stats import t
import numpy as np
import pandas as pd

def reduce_dimensions_pca(embeddings, dimensions=80):
    """ 
    Reduces the number of dimensions using PCA. 

    :embeddings: list of lists size m*n where n is the number of dimensions.
    :dimensions: number of principal components to keep.
    :return: list of lists size m*dimensions (reduced data)
    """
    pca = PCA(n_components=dimensions)
    reduced = pca.fit_transform(embeddings)
    return reduced

def reduce_dimensions_umap(embeddings, dimensions = 80, n_neighbors=10):
    """
    Uses UMAP to reduce the dimensionality of the embeddings.

    :embeddings: list of lists size m*n where n is the number of dimensions.
    :dimensions: number of components to keep.
    :return: list of lists size m*dimensions (reduced data) 
    """
    reduced = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=0.0,
        n_components=dimensions,
        random_state=42,
    ).fit_transform(embeddings)
    return reduced

def shuffle(df, n=1, axis=0):
    """
    Shuffles the data by each column or row for a pandas dataframe.

    :df: pandas dataframe shaped m*n
    :n: number of randomizations
    :axis: on which axis you want to permute. 
    :return: pandas dataframe shuffled. 
    """  
    df = df.copy()
    for _ in range(n):
        df.apply(np.random.shuffle, axis=axis)
    return df


def single_sample_t_test(sample, population_stat = 0):
    return (sample.mean() - population_stat) / (sample.std()/(len(sample)**0.5))


def calc_perm_variance(pca, embeddings_df, n_simulations=5):
    """
    Calculates the variance explained for a PCA of the permuted
    data.

    :pca: sklearn pca object
    :embeddings_df: a pandas dataframe of the embedding vectors (m*n).
    :n_simulations: integer for the number of permutations to run it on.
    :return: list of lists as a pandas dataframe with the variance explained.
    """
    pca = PCA(svd_solver='full')
    with_perm_var = []
    simulations = n_simulations
    for i in range(simulations):
        pca.fit(shuffle(embeddings_df))
        with_perm_var.append(pca.explained_variance_ratio_)
    return pd.DataFrame(with_perm_var).transpose()


def get_optimal_n_components(embeddings, n_simulations=5):
    """
    Calculates the optimal number of principal components to
    keep in a dimension reduction situation. It calculates a 
    'noise' threshold by permuting the variables to remove any correlations.
    Once that has been done, you calculate the variance explained by
    the principal components of the permuted data. 

    I then run a t test to find which principal components are significantly
    different from the baseline noise. 

    :embeddings: list of lists (m*n) of floats. where m is the number of vectors,
    and n the number of variables in each vector. 
    :n_simulations: the number of permulations to get a distribution of the noise.
    :return: integer for the optimal number of principal components. 
    """
    embeddings_df = pd.DataFrame(embeddings)
    pca = PCA(svd_solver='full')
    pca.fit(embeddings)
    without_permutation_variance = pca.explained_variance_ratio_
    with_perm_var_df = calc_perm_variance(pca, embeddings_df, n_simulations)
    pvalues = []
    for component in range(len(without_permutation_variance)):
        degrees_freedom = n_simulations - 1 # n -1
        t_stat = single_sample_t_test(with_perm_var_df.iloc[component], without_permutation_variance[component])
        p = t.cdf(t_stat,df=degrees_freedom)
        pvalues.append(p)
        if p > 0.05: break
    res = next(x for x, val in enumerate(pvalues) if val > 0.05)
    return res


def kmeans_clustering(reduced, max_num_clusters = 75):
    """
    This function calculates clusters based on the reduced vectors. 
    I also calculates the best number of clusters using the elbow method.
    max_num_clusters is the maximum number of clusters to calculate to find the optimal number. 

    :reduced: list of lists (m*n) of floats. where m is the number of vectors,
    and n the number of variables in each vector.
    :max_num_clusters: integer for the upper bound number of clusters.
    :return: list of cluster numbers for each element in reduced.
    """
    vectors = reduced

    # Calculates the inertia (cluster dispersion) for each model with i number of clusters.
    # Using k-means for the clustering. 
    # Intertia reference: https://stats.stackexchange.com/questions/78313/clustering-inertia-formula-in-scikit-learn
    inertia = []
    for i in range(1,max_num_clusters):
        kmeans = KMeans(n_clusters=i,init='k-means++', random_state=42)
        kmeans.fit(vectors)
        inertia.append(kmeans.inertia_)
    
    # Calculate the optimal number of clusters
    n_clusters = 2 
    max_relative_strength = 0
    for n in range(2,len(inertia)):
        d_0 = inertia[n-2] - inertia[n-1]
        d_1 = inertia[n-1] - inertia[n]
        d2_1 = d_0 - d_1
        s_0 = max(0,d2_1-d_1)
        relative_s = s_0 / n

        if relative_s > max_relative_strength:
            n_clusters = n
            max_relative_strength = relative_s

    # Fit the model with the optimal number of clusters. 
    kmeans = KMeans(n_clusters=n_clusters,init='k-means++', n_init=1, max_iter=200)
    labels = kmeans.fit_predict(vectors)
    return labels


def hdbscan_clustering(reduced, min_cluster_size=4):
    """
    Uses HDBSCAN to calculate clusters from the reduced data.
    :reduced: list of lists (m*n) of floats. where m is the number of vectors,
    and n the number of variables in each vector.
    :return: list of cluster numbers for each element in reduced. 
    (note that -1 is an outlier)
    """
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    cluster_labels = clusterer.fit_predict(reduced)
    return cluster_labels