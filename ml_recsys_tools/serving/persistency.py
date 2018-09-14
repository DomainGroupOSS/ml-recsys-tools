import random
import time
import functools
from threading import Thread

from flask import jsonify

from ml_recsys_tools.utils.logger import simple_logger as logger
from ml_recsys_tools.utils.dataio import S3FileIO
from ml_recsys_tools.recommenders.recommender_base import BaseDFSparseRecommender


def safe_jsonify(fn):
    @functools.wraps(fn)
    def inner(*args, **kwargs):
        try:
            return jsonify(fn(*args, **kwargs) )
        except Exception as e:
            logger.exception(e)
            return jsonify({'error': str(e)})
    return inner


class S3ModelReloaderServer:
    def __init__(self, s3_bucket, latest_path_key,
                 update_interval_hours=2,
                 interval_jitter_ratio=0,
                 block_until_first_load=False,
                 ):
        """
        :param s3_bucket: the S3 bucket used to store the models and the pointer to latest model
        :param latest_path_key: the path to an S3 pickle file that contains the S3 path to the latest model
        :param update_interval_hours: the interval for checking for updated model
            (checks whether the latest_path_key has changed)
        :param interval_jitter_ratio: ratio of jitter for the time interval
            (this is to reduce resources spike in parallel workers)
        :param block_until_first_load: whether to block execution until first model is loaded, if False,
            some methods may be broken because self.model will be None until loaded and tested
        """
        self.keep_reloading = True
        self.model = None
        self._s3_bucket = s3_bucket
        self._latest_path_key = latest_path_key
        self._current_model_path = None
        self._update_interval_seconds = update_interval_hours * 3600 * (1 - interval_jitter_ratio/2)
        self._interval_jitter_seconds = interval_jitter_ratio * self._update_interval_seconds
        self._reloader_thread = Thread(target=self._model_reloader)
        self._reloader_thread.start()
        if block_until_first_load:
            self._block_until_first_load()

    def _block_until_first_load(self):
        wait_sec = 0
        while self.keep_reloading and (self.model is None):
            if wait_sec==0 or (wait_sec % 10)==0:
                logger.info('Blocking until first model is loaded (%d seconds already).' % wait_sec)
            time.sleep(1)
            wait_sec += 1

    def __del__(self):
        self.keep_reloading = False

    def version(self):
        return {'model_version': self._current_model_path}

    def status(self):
        status = 'ready' if self.model_ready() else 'loading'
        return {'model_status': status}

    def model_ready(self):
        return self.version()['model_version'] is not None

    def _latest_s3_model_path(self):
        path = S3FileIO(self._s3_bucket).unpickle(self._latest_path_key)
        if not path:
            raise ValueError('S3 model path is empty (%s).' % str(path))
        return path

    def _test_loaded_model(self, model):
        if model is None:
            raise ValueError('Downloaded empty model')

    def _time_jitter(self):
        return random.random() * self._interval_jitter_seconds

    def _model_reloader(self):
        time.sleep(self._time_jitter())
        while self.keep_reloading:
            try:
                new_model_s3_path = self._latest_s3_model_path()
                if new_model_s3_path == self._current_model_path:
                    logger.info('Model path unchanged, not reloading. %s' % new_model_s3_path)
                else:
                    updated_model = S3FileIO(self._s3_bucket).unpickle(new_model_s3_path)
                    self._test_loaded_model(updated_model)
                    self.model = updated_model
                    self._current_model_path = new_model_s3_path
                    logger.info('Loaded updated model from S3. %s' % new_model_s3_path)
            except Exception as e:
                logger.error('Failed model update. %s' % str(e))
                logger.exception(e)
                if self.model is None:
                    raise EnvironmentError('Could not load model on startup.')
            time.sleep(self._update_interval_seconds + self._time_jitter())


class RankingModelServer(S3ModelReloaderServer):

    mode_default = 'default'
    mode_resort = 'resort'
    mode_adjust = 'adjust'
    mode_combine = 'combine'
    mode_disabled = 'disabled'

    def _test_loaded_model(self, model):
        super()._test_loaded_model(model)
        # tests and warms up model (first inference is a bit slower for some reason)
        _ = self._rank_items_for_user(
            model, user_id='test_user',
            item_ids=['test_item1', 'test_item2'], mode=self.mode_resort)

    @classmethod
    def _combine_original_order(cls, mode):
        return mode in [cls.mode_adjust, cls.mode_combine]

    @classmethod
    def _rank_items_for_user(cls, model: BaseDFSparseRecommender,
                             user_id, item_ids, mode, min_score=None):
        ts = time.time()

        if mode==cls.mode_disabled:
            scores = [None] * len(item_ids)
        else:
            pred_df = model.predict_for_user(
                user_id=user_id,
                item_ids=item_ids,
                rank_training_last=True,
                sort=True,
                combine_original_order=cls._combine_original_order(mode),
                )

            item_ids = pred_df[model._item_col].tolist()

            scores = pred_df[model._prediction_col].values

            if min_score is not None:
                unknowns_mask = scores < min_score
                n_unknowns = unknowns_mask.sum()  # is a numpy array
                scores[unknowns_mask] = min_score
            else:
                n_unknowns = 0

            scores = scores.tolist()

        result = {'user_id': user_id, 'ranked_items': item_ids, 'scores': scores}

        logger.info('Ran ranking for user %s (%d items, %d unknown) in %.3f seconds for mode %s.' %
                    (str(user_id), len(scores), n_unknowns, time.time() - ts, str(mode)))
        return result

    def rank_items_for_user(self, user_id, item_ids, mode, min_score=None):
        return self._rank_items_for_user(
            model=self.model,
            user_id=user_id,
            item_ids=item_ids,
            mode=mode,
            min_score=min_score)

    def unknown_users(self, user_ids):
        mask = self.model.unknown_users_mask(user_ids).tolist()
        return [u for u, m in zip(user_ids, mask) if m]

    def unknown_items(self, item_ids):
        mask = self.model.unknown_items_mask(item_ids).tolist()
        return [i for i, m in zip(item_ids, mask) if m]



