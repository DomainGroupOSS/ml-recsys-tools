import json
import pickle
import boto3
import redis
from ml_recsys_tools.utils.logger import simple_logger as logger


class RedisTable(redis.StrictRedis):

    def __init__(self, host_url, table_name, timeout=10, *args, **kwargs):
        self.table_name = table_name
        super().__init__(
            host=host_url, port=6379, decode_responses=True,
            socket_timeout=timeout, socket_connect_timeout=timeout,
            *args, **kwargs)

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
        response = self.get(key)
        if response:
            try:
                response = json.loads(response)
            except Exception as e:
                logger.exception(e)
                logger.error('Failed redis query. key: %s, response: %s' %
                             (key, response))
        return response


class S3FileIO:
    def __init__(self, bucket_name):
        self.bucket_name = bucket_name

    def write(self, data, remote_path):
        try:
            client = boto3.client('s3')
            resp = client.put_object(
                Body=data,
                Bucket=self.bucket_name,
                Key=remote_path)
            return resp['ResponseMetadata']['HTTPStatusCode']==200
        except Exception as e:
            logger.error('Failed writing to S3: %s' % str(e))
            logger.exception(e)

    def read(self, remote_path):
        try:
            client = boto3.client('s3')
            resp = client.get_object(
                Bucket=self.bucket_name,
                Key=remote_path)
            return resp['Body'].read()
        except Exception as e:
            logger.error('Failed reading from S3: %s' % str(e))
            logger.exception(e)

    def pickle(self, obj, remote_path):
        logger.info('S3: pickling to %s' % remote_path)
        return self.write(pickle.dumps(obj), remote_path)

    def unpickle(self, remote_path):
        logger.info('S3: unpickling from %s' % remote_path)
        return pickle.loads(self.read(remote_path))

    def remote_to_local(self, remote_path, local_path):
        logger.info('S3: copying from %s to %s' % (remote_path, local_path))
        with open(local_path, 'wb') as local:
            local.write(self.read(remote_path))

    def local_to_remote(self, local_path, remote_path):
        logger.info('S3: copying from %s to %s' % (local_path, remote_path))
        with open(local_path, 'rb') as local:
            self.write(local.read(), remote_path)

