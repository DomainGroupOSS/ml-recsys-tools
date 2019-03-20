import numpy as np
import pandas as pd

from ml_recsys_tools.data_handlers.interactions_with_features import ItemsHandler, ObsWithFeatures
from ml_recsys_tools.recommenders.recommender_base import BaseDFSparseRecommender
from ml_recsys_tools.utils.geo import ItemsGeoMap, PropertyGeoMap


class ItemsGeoMapper:

    score_col = '_score'

    def __init__(self, items_handler: ItemsHandler, map: ItemsGeoMap):
        self.map = map
        self.items_handler = items_handler

    def join_scores_list(self, items_df, item_ids, scores):
        return items_df. \
            set_index(self.items_handler.item_id_col). \
            join(pd.Series(scores, index=item_ids, name=self.score_col)). \
            reset_index().\
            sort_values(self.score_col, ascending=False)

    def _scale_scores_as_marker_sizes(self, scores, min_size=5):
        scale_scores = np.array(scores) ** 2
        scale_scores -= scale_scores.min()
        scale_scores *= 10 / scale_scores.max()
        scale_scores += min_size
        scale_scores = [int(v) for v in scale_scores]
        return scale_scores

    def _assign_data_to_map(self, df):
        df = df.set_index(self.items_handler.item_id_col).drop_duplicates()
        self.map.df_items = df

    def map_recommendations(self, train_ids, reco_ids, scores,
                            test_ids=(), **kwargs):
        return self.map_items_and_scores(
            source_ids=train_ids,
            result_ids=reco_ids,
            test_ids=test_ids,
            result_scores=scores,
            **kwargs)

    def map_similar_items(self, source_id, similar_ids, scores, **kwargs):
        return self.map_items_and_scores(
            source_ids=[source_id],
            result_ids=similar_ids,
            result_scores=scores,
            **kwargs)

    def map_items_and_scores(self, source_ids, result_ids, result_scores,
                             test_ids=(), color=(210, 20, 210), marker_size=5):
        return self.map_recommendation_variants(
            train_ids=source_ids,
            test_ids=test_ids,
            recs_variants=[result_ids],
            scores_lists=[result_scores],
            colors=[color],
            marker_size=marker_size)

    def map_recommendation_variants(
            self, train_ids, test_ids, recs_variants,
            colors=None, scores_lists=None, marker_size=5):

        self.map.reset_map()
        train_items = self.items_handler.items_filtered_by_ids(train_ids)
        test_items = self.items_handler.items_filtered_by_ids(test_ids)
        rec_items_dfs = [self.items_handler.items_filtered_by_ids(rec) for rec in recs_variants]

        all_data = pd.concat(rec_items_dfs + [train_items, test_items], sort=False)
        self._assign_data_to_map(all_data)

        if colors is None:
            scores_lists = [None] * len(recs_variants)

        if colors is None:
            colors = self.map.get_n_spaced_colors(len(recs_variants))

        for rec_items, rec_ids, scores, color in zip(rec_items_dfs, recs_variants, scores_lists, colors):

            scale_scores = None
            if scores is not None:

                rec_items = self.join_scores_list(rec_items,
                                                  item_ids=rec_ids,
                                                  scores=scores)

                scale_scores = self._scale_scores_as_marker_sizes(
                    scores=rec_items[self.score_col].fillna(0).values, min_size=marker_size)

            self.map.add_markers(rec_items, color=color, size=scale_scores or marker_size, fill=False)
            self.map.add_heatmap(rec_items, color=color, sensitivity=1, opacity=0.4, spread=50)

        if len(train_items):
            self.map.add_markers(train_items, color='black', size=6)

        if len(test_items):
            self.map.add_markers(test_items, color='gray', size=6)

        return self.map


class ObsItemsGeoMapper(ItemsGeoMapper):

    def __init__(self, obs_handler: ObsWithFeatures, map: ItemsGeoMap):
        super().__init__(map=map, items_handler=obs_handler)
        self.obs_handler = obs_handler

    def map_items_for_user(self, user):
        self.map.df_items = self.obs_handler.get_items_df_for_user(user)
        return self.map

    def map_items_by_common_items(self, item_id, default_marker_size=2):
        self.map.reset_map()

        users = self.obs_handler.items_filtered_df_obs(item_id)[self.obs_handler.uid_col].unique().tolist()

        items_dfs = [self.obs_handler.get_items_df_for_user(user) for user in users]

        # unite all data and get counts
        all_data = pd.concat(items_dfs, sort=False)
        counts = all_data[self.obs_handler.item_id_col].value_counts()
        all_data = all_data.set_index(self.obs_handler.item_id_col).drop_duplicates()
        all_data['counts'] = counts
        all_data.reset_index(inplace=True)

        self._assign_data_to_map(all_data)

        # add maps for each user's history
        colors = iter(self.map.get_n_spaced_colors(len(items_dfs)))
        for df in items_dfs:
            color = next(colors)
            self.map.add_markers(df, color=color, size=default_marker_size)
            self.map.add_heatmap(df, color=color, sensitivity=1, opacity=0.4)

        # add common items
        common_items = all_data[all_data['counts'].values > 1]
        sizes = list(map(int, np.sqrt(common_items['counts'].values) + default_marker_size))
        self.map.add_markers(common_items, color='white', size=sizes)

        return self.map

    def map_cluster_labels(self, df_items=None, sample_n=1000):
        self.map.reset_map()

        if df_items is None:
            df_items = self.obs_handler.df_items

        unique_labels = df_items[self.obs_handler.cluster_label_col].unique()

        items_dfs = [df_items[df_items[self.obs_handler.cluster_label_col] == label].sample(sample_n)
                     for label in unique_labels]

        self._assign_data_to_map(pd.concat(items_dfs, sort=False))

        colors = iter(self.map.get_n_spaced_colors(len(items_dfs)))

        for df in items_dfs:
            color = next(colors)
            self.map.add_heatmap(df, color=color, spread=50, sensitivity=3, opacity=0.3)

        return self.map


class RecommenderGeoVisualiser:

    default_height = 800

    def __init__(self,
                 recommender: BaseDFSparseRecommender,
                 items_handler: ItemsHandler,
                 link_base_url='www.domain.com.au'):
        self.recommender = recommender
        self.items_handler = items_handler
        self.mapper = ItemsGeoMapper(
            items_handler=self.items_handler,
            map=PropertyGeoMap(link_base_url=link_base_url))

    def random_user(self):
        return np.random.choice(self.recommender.all_users)

    def random_item(self):
        return np.random.choice(self.recommender.all_items)

    def _embed_data_in_html_template(
            self, meta_data, source_items, result_items, result_scores):

        source_df = self.items_handler.items_filtered_by_ids(source_items)
        result_df = self.items_handler.items_filtered_by_ids(result_items)
        result_df = self.mapper.join_scores_list(items_df=result_df,
                                               item_ids=result_items,
                                               scores=result_scores)

        metadata_str = str(meta_data).\
            replace('{','').replace('}','').replace("'",'').\
            replace(': ','=').replace(', ','&')

        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>{{title}}</title>
        </head>
        <body>
        <p>Metadata: {metadata_str}</p>
        {{snippet}}
        <p>Source data:</p>
        {source_df.to_html()}
        <p>Result data:</p>
        {result_df.to_html()}
        </body>
        </html>
        """
        return html_template

    def _scale_scores(self, scores):
        scores = np.array(scores)
        scores -= scores.min()
        scores /= scores.max()
        return scores

    def _score_cutoff_ind(self, scores, score_cutoff):
        scores = self._scale_scores(scores)
        return np.min(np.argwhere(scores < score_cutoff))

    def _user_recommendations_and_scores(self, user):
        recos = self.recommender.get_recommendations([user])
        reco_items = np.array(recos[self.recommender._item_col].values[0])
        reco_scores = np.array(recos[self.recommender._prediction_col].values[0])
        reco_scores /= reco_scores.max()
        return reco_items, reco_scores

    def _similar_items_and_scores(self, item):
        if hasattr(self.recommender, 'get_similar_items'):
            simils = self.recommender.get_similar_items([item])
            simil_items = np.array(simils[self.recommender._item_col].values[0])
            simil_scores = np.array(simils[self.recommender._prediction_col].values[0])
            simil_scores /= simil_scores.max()
            return simil_items, simil_scores
        else:
            raise NotImplementedError(f"'get_similar_items' is not implemented / supported by "
                                      f"{self.recommender.__class__.__name__}")

    def _user_training_items(self, user):
        known_users = np.array([user])[~self.recommender.unknown_users_mask([user])]
        user_ind = self.recommender.user_inds(known_users)
        training_items = self.recommender.item_ids(
            self.recommender.train_mat[user_ind, :].indices).astype(str)
        return training_items

    def map_recommendations(self, user, *, path=None, score_cutoff=0):
        training_items = self._user_training_items(user)
        reco_items, reco_scores = self._user_recommendations_and_scores(user)
        return self.map_for_items_and_scores(
            source_items=training_items,
            result_items=reco_items,
            result_scores=reco_scores,
            metadata={'user': user},
            path=path,
            score_cutoff=score_cutoff,
            title=f'user: {user}')

    def map_similar_items(self, item, *, path=None, score_cutoff=0):
        simil_items, simil_scores = self._similar_items_and_scores(item)
        return self.map_for_items_and_scores(
            source_items=[item],
            result_items=simil_items,
            result_scores=simil_scores,
            metadata={'item': item},
            path=path,
            score_cutoff=score_cutoff,
            title=f'item: {item}')

    def map_for_items_and_scores(self, source_items, result_items, result_scores,
                                 title='', metadata=None, path=None, score_cutoff=0):
        if score_cutoff:
            ind = self._score_cutoff_ind(result_scores, score_cutoff=score_cutoff)
            result_items = result_items[:ind]
            result_scores = result_scores[:ind]

        if result_items:
            html_template = self._embed_data_in_html_template(meta_data=metadata,
                                                              source_items=source_items,
                                                              result_items=result_items,
                                                              result_scores=result_scores)
            map_obj = self.mapper.map_items_and_scores(
                source_ids=source_items,
                result_ids=result_items,
                result_scores=result_scores
            )
            return map_obj.write_html(path=path, title=title,
                                      template=html_template, map_height=self.default_height)
        else:
            return f'No items higher than cutoff score {score_cutoff}'