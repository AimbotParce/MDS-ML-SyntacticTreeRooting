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