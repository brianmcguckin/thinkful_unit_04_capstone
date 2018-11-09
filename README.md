# Reuters-21578 Text Classification

[Jupyter notebook](04_capstone_unsupervised_learning_final.ipynb)<br>

**Toolkit:**
- Python 3, numpy, pandas, matplotlib, seaborn, scipy, sklearn, xgboost, nltk, re, umap

**Topics:**
- Text (article) classification using the Reuters-21578 dataset
- Text cleaning, tokenization, vectorization (tf-idf)
- Dimensionality reduction: truncated singular value decomposition (latent semantic analysis)
- Clustering: k-means (& minibatch), spectral clustering, mean-shift, affinity propagation
- Supervised classification models: KNN, logistic regression, random forest, gradient boosted trees (xgboost)
- Evaluation: ground truth, ARI (adjusted rand index), cross validation

# 1. Introduction
- Project goal: utilize unsupervised machine learning algorithms to categorize article topics based on their text
- Dataset: Reuters-21578 from NLTK corpus
  - Pre-labeled articles from 1987
  - 90 distinct categories, articles can have more than one category label
  - Focused on articles with a single label
  - Trained & tested algorithms on binary classes as well as a three category classification

# 2. Data Preprocessing
- Text cleaning: remove special characters and digits, lowercase text
- Parse into tokens using spacy
- Feature generation: tf-idf vectorization
- Dimensionality reduction:
  - LSA (Latent Semantic Analysis): truncated SVD method in SKLearn
  - UMAP (Uniform Manifold Approximation & Projection): non-linear dimension reduction technique

# 3. Modeling
- **Unsupervised Classification (Clustering)**: (somewhat expected) poor performance, visualizing ground truth clusters suggested that topics may not be very separable
- **Evaluation**: Adjusted Rand Index (ARI), ground truth
- Ground Truth:
![ground_truth](https://raw.githubusercontent.com/brianmcguckin/thinkful_unit_04_capstone/master/images/ground_truth.png "ground_truth.png")

- K-Means:
![kmeans_clusters](https://raw.githubusercontent.com/brianmcguckin/thinkful_unit_04_capstone/master/images/kmeans_clusters.png "kmeans_clusters.png")

- Spectral Clustering:
![spectral_knn_clusters](https://raw.githubusercontent.com/brianmcguckin/thinkful_unit_04_capstone/master/images/spectral_knn_clusters.png "spectral_knn_clusters.png")

- Mean Shift:
![mean_shift](https://raw.githubusercontent.com/brianmcguckin/thinkful_unit_04_capstone/master/images/mean_shift_clusters.png "mean_shift_clusters.png")

- **Supervised Classification**: much better performance, even before hyperparameter tuning, supervised classifiers did quite while (accuracy in the low to mid .9 range)
- **Evaluation**: mean accuracy (10 fold cross validation)
- Models:
  - KNN: train 0.9032, test 0.9459
  - Logistic Regression: train 0.9777, test 0.9633
  - Random Forest: train 0.9655, test 0.9492
  - XGBoost: train 0.9755, test 0.9659
