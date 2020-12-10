# document-vector-clustering

This is a [clustering](https://en.wikipedia.org/wiki/Cluster_analysis) library to take [document vectors](https://towardsdatascience.com/document-embedding-techniques-fed3e7a6a25d) and cluster them based on [semantic similarity](https://en.wikipedia.org/wiki/Semantic_similarity#:~:text=Semantic%20similarity%20is%20a%20metric,as%20opposed%20to%20lexicographical%20similarity.). With this library, you can get 'topic' labels for each of the documents. This allows you to identify documents that are similar (i.e. group like with like).

## Dimension Reduction
Due to the usual [high-dimensionality](https://en.wikipedia.org/wiki/Dimension#Additional_dimensions) of document vectors, one needs to [reduce the number of dimensions](https://en.wikipedia.org/wiki/Dimensionality_reduction#:~:text=Dimensionality%20reduction%2C%20or%20dimension%20reduction,close%20to%20its%20intrinsic%20dimension.) of the vectors to a more manageable size without losing too much of the variation in the data. If the number of dimensions is too great, you can get the '[curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality)' problem. When you get this problem, the the volume of the space increases so fast that the available data become sparse. As a result, clustering becomes pointless. 

As a result, I have made it possible to apply [Principal Component Analysis](https://en.wikipedia.org/wiki/Principal_component_analysis#:~:text=Principal%20component%20analysis%20(PCA)%20is,components%20and%20ignoring%20the%20rest.) (PCA) and [Uniform Manifold Approximation and Projection](https://arxiv.org/abs/1802.03426) (UMAP) for dimensionality reduction. PCA is the standard method for dimensionality reduction, however, [UMAP does pretty well](https://towardsdatascience.com/tsne-vs-umap-global-structure-4d8045acba17#:~:text=UMAP%20for%20Non%2DLinear%20Manifold&text=This%20is%20because%20PCA%20as,2D%20dimensionality%20of%20the%20data.).  

## Clustering
For clustering, I provide two methods:
1. [k-means](https://en.wikipedia.org/wiki/K-means_clustering)
2. [HDBSCAN](https://en.wikipedia.org/wiki/DBSCAN)

From experience, k-means typically does pretty well with few documents and fewer than 15 dimensions. On bigger data with unknown number of topics, HDBSCAN comes to the rescue.  

Note: The k-means function, kmeans_clustering, finds the optimal number of clusters k by using the ['elbow' method](https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/). 

# Installation

First, clone the repository. 

In your terminal, run:  
```
pip install -r requirements.txt
```

# Basic Usage

```python
>>> import clustering
>>> reduced = clustering.reduce_dimensions_umap(document_embeddings)
>>> document_labels = clustering.hdbscan_clustering(reduced)
```

# Other Functionality

## Finding the optimal number of principal components to keep
When reducing the number of dimensions in the document/sentence vectors, there is not a lot of guidance with regards to how many principal components to keep. Ideally, you would be able to choose a number where you keep all the components that explain a 'significant' amount of variation.  

So I have used [this method](https://towardsdatascience.com/how-to-tune-hyperparameters-of-tsne-7c0596a18868) to determine how many components to keep. Basically, you shuffle the data by columns to remove any correlations in the data. You then run a PCA on the shuffled data to get an average amount of variation explained by the principal components. The variation explained becomes a 'noise' baseline. With several shuffles of the data, you can get a [spread](https://en.wikipedia.org/wiki/Statistical_dispersion) of the 'noise'. When running the PCA on the original data, you can perform a [t-test](https://en.wikipedia.org/wiki/Student%27s_t-test) to determine what principal components explain a statistically significant variation in the data. 