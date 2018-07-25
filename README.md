# ml-recsys-tools

#### Open source repo for various tools for recommender systems development work. Work in progress.

#### Main purpose is to provide a single wrapper for various recommender packages to train, tune, evaluate and get data in and recommendations / similarities out.

#### Installation:
Pip: includes only basic dependecies + lightfm/implicit for now: `pip install ml_recsys_tools`

Docker (build and run interactively): 
```
docker build -t ml_recsys_tools:local .
docker run -it --rm ml_recsys_tools:local python
```     

#### Basic example:
    # dataset: download and prepare dataframes
    from ml_recsys_tools.datasets.prep_movielense_data import get_and_prep_data
    rating_csv_path, users_csv_path, movies_csv_path = get_and_prep_data()
    
    # read the interactions dataframe and create a data handler object and  split to train and test
    import pandas as pd
    
    ratings_df = pd.read_csv(rating_csv_path)
    from ml_recsys_tools.data_handlers.interaction_handlers_base import ObservationsDF    
    obs = ObservationsDF(ratings_df, uid_col='userid', iid_col='itemid')
    train_obs, test_obs = obs.split_train_test(ratio=0.2)
    
    # train and test LightFM recommender
    from ml_recsys_tools.recommenders.lightfm_recommender import LightFMRecommender    
    lfm_rec = LightFMRecommender()
    lfm_rec.fit(train_obs, epochs=10)
    
    # print summary evaluation report:
    print(lfm_rec.eval_on_test_by_ranking(test_obs.df_obs, prefix='lfm ', n_rec=100))
    
    # get all recommendations and print a sample (training interactions are filtered out by default)
    recs = lfm_rec.get_recommendations(lfm_rec.all_users, n_rec=5)
    print(recs.sample(5))
    
    # get all similarities and print a sample
    simils = lfm_rec.get_similar_items(lfm_rec.all_items, n_simil=5)
    print(simils.sample(10))


## Recommender models and tools:

* #### [LightFM](https://github.com/lyst/lightfm) package based recommender.
* #### [Spotlight](https://github.com/maciejkula/spotlight) package based implicit recommender.
* #### [Implicit](https://github.com/benfred/implicit) package based ALS recommender.
* #### Serving / Tuning / Evaluation features added for most recommenders:
    * Dataframes for all inputs and outputs
        * adding external features (for LightFM hybrid mode)
        * early stopping fit (for iterative models: LightFM, ALS, Spotlight)
        * hyperparameter search
        * fast batched methods for:
            * user recommendation sampling
            * similar items samplilng with different similarity measures
            * similar users sampling
            * evaluation by sampling and ranking      
                  
* #### Additional recommender models:
    * ##### Similarity based:
        * cooccurence (items, users)
        * generic similarity based (can be used with external features)  
              
* #### Ensembles:
    * subdivision based (multiple recommenders each on subset of data - e.g. geographical region):
        * geo based: simple grid, equidense grid, geo clustering
        * LightFM and cooccurrence based
    * combination based - combining recommendations from multiple recommenders
    * similarity combination based - similarity based recommender on similarities from multiple recommenders
    * cascade ensemble 
           
* #### Interaction dataframe and sparse matrix handlers / builders:
    * sampling, data splitting,
    * external features matrix creation (additional item features),
        with feature engineering: binning / one*hot encoding (via pandas_sklearn)
    * evaluation and ranking helpers
    * handlers for observations coupled with external features and features with geo coordinates
    * mappers for geo features, observations, recommendations, similarities etc.
        
* #### Evaluation utils:
    * score reports on lightfm metrics (AUC, precision, recall, reciprocal)
    * n-DCG, and n-MRR metrics, n-precision / recall
    * references: best possible ranking and chance ranking

* #### Utilities:
    * hyperparameter tuning utils (by skopt)
    * similarity calculation helpers (similarities, dot, top N, top N on sparse)
    * parallelism utils
    * sklearn transformer extenstions (for feature engineering)
    * google maps util for displaying geographical data
    * logging, debug printouts decorators and other isntrumentation and inspection tools
    * pandas utils
    * data helpers: redis, s3    
    * basic rankings server that updates models from S3 
 

* #### Examples:
    * a basic example on movielens 1M demonstrating:
        * basic data ingestion without any item/user features
        * LightFM recommender:
            fit, evaluation, early stopping,
            hyper-param search, recommendations, similarities
        * Cooccurrence recommender
        * Two combination ensembles (Ranks and Simils)

* #### Still to add:
    * add and reorganize examples 
    * much more comments and docstrings
    * more tests

