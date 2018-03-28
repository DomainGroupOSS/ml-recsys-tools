# ml-recsys-tools

Open source repo for various tools for recommender systems development work.
Includes:

Recommender model and LightFM tools:
    - LightFM and other recommenders with:
            - dataframes for all inputs and outputs
            - adding external features
            - early stopping fit
            - hyperparam search
            - fast batched user recommendation sampling
            - fast batched similar items samplilng with different similarity measures
            - fast batched similar users sampling
            - fast evaluation by sampling and ranking
    - additional models:
        - similarity based recommenders:
            - cooccurence (items, users)
            - generic similarity based
        - ensembles:
            - subdivision based (multiple recommenders each on subset of data - e.g. geographical region)
            - combination based - combining recommendations from multiple recommenders
            - similarity combination based - similarity based recommender on similarities from multiple recommenders
            - cascade ensemble

  - interaction dataframe and sparse matrix handlers / builders:
    - sampling, data splitting,
    - external features matrix creation (additional item features),
        with feature engineering: binning / one-hot encoding (via pandas_sklearn)
    - evaluation and ranking helpers

  - evaluation utils:
    - score reports on lightfm metrics (AUC, precision, recall, reciprocal)
    - n-DCG, and n-MRR metrics, n-precision / recall
    - references: best possible ranking and chance ranking

Utilities:
    - hyperparameter tuning utils (by skopt)
    - similarity calculation helpers (similarities, dot, top N, top N on sparse)
    - parallelism utils
    - sklearn transformer extenstions (for feature engineering)
    - google maps util for displaying geographical data
    - logging, debug printouts decorators
    - pandas utils


Still to add:
- examples
- comments and documentation
- tests
 
