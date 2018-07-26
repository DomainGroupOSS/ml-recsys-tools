"""
This is an example on datasets-1M demonstrating recommenders from spotlight library
"""

from ml_recsys_tools.datasets.prep_movielense_data import get_and_prep_data
import pandas as pd
from ml_recsys_tools.data_handlers.interaction_handlers_base import ObservationsDF

rating_csv_path, users_csv_path, movies_csv_path = get_and_prep_data()
ratings_df = pd.read_csv(rating_csv_path)

obs = ObservationsDF(ratings_df, uid_col='userid', iid_col='itemid', timestamp_col='timestamp')
train_obs, test_obs = obs.split_train_test(ratio=0.2, time_split_column=obs.timestamp_col)


from ml_recsys_tools.recommenders.spotlight_recommenders import EmbeddingFactorsRecommender
emb_rec = EmbeddingFactorsRecommender(model_params=dict(loss='adaptive_hinge', n_iter=1))
# emb_rec.fit(train_obs)
emb_rec.fit_with_early_stop(train_obs, epochs_max=5, epochs_step=1)
print(emb_rec.eval_on_test_by_ranking(test_obs, prefix='implicit embeddings '))


# trying to reproduce this:
# https://github.com/maciejkula/spotlight/tree/master/examples/movielens_sequence
from ml_recsys_tools.recommenders.spotlight_recommenders import SequenceEmbeddingRecommender
seq_rec = SequenceEmbeddingRecommender(
    model_params=dict(n_iter=15, embedding_dim=32, batch_size=32, learning_rate=0.01),
    fit_params=dict(max_sequence_length=200, timestamp_col='timestamp'))
seq_rec.fit(train_obs)
# emb_rec.fit_with_early_stop(train_obs, epochs_max=30, epochs_step=3)
print(seq_rec.eval_on_test_by_ranking(test_obs, prefix='lstm ', include_train=False))

# same experiment: CNNs
from ml_recsys_tools.recommenders.spotlight_recommenders import CNNEmbeddingRecommender
cnn_rec = CNNEmbeddingRecommender(
    fit_params=dict(max_sequence_length=200, timestamp_col='timestamp'))
cnn_rec.fit(train_obs)
print(cnn_rec.eval_on_test_by_ranking(test_obs, prefix='cnn ', include_train=False))
