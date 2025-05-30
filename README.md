# MDS-ML-SyntacticTreeRooting

Given a syntactic dependency tree, the most important word (the root) should be a central one. This project aims at predicting the root word from syntactic free trees


## Files available

In the [`data/`](data/) folder, two datasets are provided, a [`train.csv`](data/train.csv) file containing the training
sentences with their root, and a [`test.csv`](data/test.csv) file containing the test sentences, the root of which is to
be predicted and submitted for evaluation. The training set contains 500 sentences in 21 languages each, while the test
set contains 495 sentences in 21 languages each. The data sets are taken from a parallel corpus, meaning that the same
sentences are present in all languages.


## Columns

* `id` - An id of the sentence, for scoring purposes in the submission file (for the test set only)
* `language` - Language of sentence
* `sentence` - Sentence number, relating sentences across languages (parallel corpus)
* `n` - Length of the sentence (or number of nodes in the tree)
* `edgelist` - List of edges representing the syntactic dependence tree
* `root` - Vertex id of the root node (for the training set only)

## Installation

Before running any code, make sure to install the required packages. You can do this by running:

```bash
pip install -r requirements.txt
```

## Usage

To train and evaluate the final model, you should run one by one all the cells in the [`src/final-model-qda.ipynb`](src/final-model-qda.ipynb) notebook.

However, if you want to perform all the tests we did, you should do so by following the steps below:
1. Run all the cells in [the baseline notebook](src/baseline-binary-classification.ipynb) to get the baseline results.
2. Run all the cells in [the feature engineering notebook](src/feature-engineering.ipynb) to compute the new features and save them to the `data/cache/` folder.
3. For each model that you want to test, run its corresponding notebook from the `src/` folder:
    - [Logistic Regression](src/logistic-regression.ipynb)
    - [LDA, QDA and NB](src/bayes_models.ipynb)
    - [Random Forest](src/random-forests.ipynb)
    - [XGBoost](src/xgboost.ipynb)
    - [KNN](src/knn.ipynb)
    - [MLP](src/mlp.ipynb) (To run the cross-validation, you will need to switch to the script `src/run_mlp.py`)
    - [Graph SAGE](src/graph-sage.ipynb) (An attempt we made to use graph neural networks, but it did not work, and took advantage of the order of the nodes in the tree)
4. Finally, run the [final model notebook](src/final-model-qda.ipynb) to train and evaluate the final model.

> Note: There's an additional notebook in the `src/` folder, [draw-graphs.ipynb](src/draw-graphs.ipynb), which we used to create and visualize the figures from the report.

