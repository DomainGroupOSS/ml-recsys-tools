# ml-recsys-tools

```
Open source repo for various tools for recommender systems development work. Work in progress.

Includes:

Recommender model and LightFM tools:

    - LightFM package based recommenders (https://github.com/lyst/lightfm).

    - Spotlight package based implicit recommender (https://github.com/maciejkula/spotlight).

    - Implicit package based ALS recommender (https://github.com/benfred/implicit).

    - Serving / Tuning / Evaluation features added for various recommenders:
            - dataframes for all inputs and outputs
            - adding external features (for LightFM hybrid mode)
            - early stopping fit (for iterative models: LightFM, ALS, Spotlight)
            - hyperparam search
            - fast batched methods for (except Spotlight for now):
                - user recommendation sampling
                - similar items samplilng with different similarity measures
                - similar users sampling
                - evaluation by sampling and ranking

    - Additional recommender models:
        - similarity based:
            - cooccurence (items, users)
            - generic similarity based (can be used for with external features)
        - ensembles:
            - subdivision based (multiple recommenders each on subset of data - e.g. geographical region):
                - geo based: simple grid, equidense grid, geo clustering
                - LightFM and cooccurrence based
            - combination based - combining recommendations from multiple recommenders
            - similarity combination based - similarity based recommender on similarities from multiple recommenders
            - cascade ensemble

    - interaction dataframe and sparse matrix handlers / builders:
        - sampling, data splitting,
        - external features matrix creation (additional item features),
            with feature engineering: binning / one-hot encoding (via pandas_sklearn)
        - evaluation and ranking helpers
        - handlers for observations coupled with external features and features with geo coordinates
        - mappers for geo features, observations, recommendations, similarities etc.

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
    - logging, debug printouts decorators and other isntrumentation and inspection tools
    - pandas utils
    - data helpers: redis, s3

Examples:
    - a basic example on movielens 1M demonstrating:
        - basic data ingestion without any item/user features
        - LightFM recommender:
            fit, evaluation, early stopping,
            hyper-param search, recommendations, similarities
        - Cooccurrence recommender
        - Two combination ensembles (Ranks and Simils)

Still to add:
- more examples
- much more comments and docstrings
- more tests

 ```
