import gzip
import io
import json
import pickle
import boto3
import redis
from ml_recsys_tools.utils.logger import simple_logger as logger


class RedisTable(redis.StrictRedis):

    def __init__(self, host_url, table_name, timeout=10, **kwargs):
        self.table_name = table_name
        super().__init__(
            host=host_url, port=6379, decode_responses=False,
            socket_timeout=timeout, socket_connect_timeout=timeout,
            **kwargs)

    @staticmethod
    def _encode_data(data, compress):
        if data:
            if compress:
                compress = compress if isinstance(compress, int) and 1<=compress<=9 else 1
                data = gzip.compress(json.dumps(data).encode(), compress)
            else:
                data = json.dumps(data)
        return data

    def set_json(self, key_name, key_value, data, compress=False, **kwargs):
        return super().set(
            self.table_index_key(key_name, key_value),
            self._encode_data(data, compress),
            **kwargs)

    def set_json_to_pipeline(
            self, pipeline, key_name, key_value, data, compress=False, **kwargs):
        return pipeline.set(
            self.table_index_key(key_name, key_value),
            self._encode_data(data, compress),
            **kwargs)

    def table_index_key(self, key, value):
        return self.table_name + ':' + key + ':' + value

    def query(self, index_key, index_value):
        """
        example:
            table = RedisTable('p-bla-bla')
            result_dict = table.query('uuid', '1234')

        :returns response JSON as dict, None if not found, or response string if JSON conversion fails
        """
        key = self.table_index_key(index_key, index_value)
        data = self.get(key)
        if data:
            try:
                data = gzip.decompress(data)
            except (OSError, TypeError):
                pass  # not gzip compressed
            try:
                data = json.loads(data)
            except Exception as e:
                logger.exception(e)
                logger.error('Failed unpacking redis query. key: %s, response: %s' %
                             (key, data))
        return data


class S3FileIO:
    def __init__(self, bucket_name):
        self.bucket_name = bucket_name

    def write_binary(self, data, remote_path, compress=True):
        try:
            client = boto3.client('s3')
            if compress:
                try:
                    # https://stackoverflow.com/questions/33562394/gzip-raised-overflowerror-size-does-not-fit-in-an-unsigned-int
                    data = gzip.compress(data, 1)
                except OverflowError:
                    pass
            with io.BytesIO(data) as f:
                client.upload_fileobj(
                    Fileobj=f,
                    Bucket=self.bucket_name,
                    Key=remote_path)
        except Exception as e:
            logger.error('Failed writing to S3: %s' % str(e))
            logger.exception(e)

    def read(self, remote_path):
        try:
            client = boto3.client('s3')
            ## for some reason this returns empty sometimes, but get_object works..
            # with io.BytesIO() as f:
            #     client.download_fileobj(
            #         Bucket=self.bucket_name,
            #         Key=remote_path,
            #         Fileobj=f)
            #     data = f.read()
            data = client.get_object(
                Bucket=self.bucket_name, Key=remote_path)['Body'].\
                read()
            try:
                data = gzip.decompress(data)
            except OSError:
                pass
            return data
        except Exception as e:
            logger.error('Failed reading from S3: %s' % str(e))
            logger.exception(e)

    def pickle(self, obj, remote_path):
        logger.info('S3: pickling to %s' % remote_path)
        return self.write_binary(pickle.dumps(obj), remote_path)

    def unpickle(self, remote_path):
        logger.info('S3: unpickling from %s' % remote_path)
        return pickle.loads(self.read(remote_path))

    def local_to_remote(self, local_path, remote_path):
        logger.info('S3: copying from %s to %s' % (local_path, remote_path))
        with open(local_path, 'rb') as local:
            self.write_binary(local.read(), remote_path)

    def remote_to_local(self, remote_path, local_path):
        logger.info('S3: copying from %s to %s' % (remote_path, local_path))
        with open(local_path, 'wb') as local:
            local.write(self.read(remote_path))

