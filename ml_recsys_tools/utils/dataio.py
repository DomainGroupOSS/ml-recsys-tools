import asyncio
import functools
import gzip
import io
import json
import logging
import os
import pickle
import smtplib
import tempfile
import time
from abc import abstractmethod
from hashlib import md5
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

import asyncpg
import boto3
import pandas as pd
import pandas.io.common
import redis
from botocore.exceptions import ClientError

from ml_recsys_tools.utils.instrumentation import log_errors, LogCallsTimeAndOutput
from ml_recsys_tools.utils.logger import simple_logger as logger

NA_VALUES = pandas.io.common._NA_VALUES.difference(['null', 'NULL'])


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

    @staticmethod
    def _decode_data(data, key):
        if data:
            try:
                data = gzip.decompress(data)
            except (OSError, TypeError):
                pass  # not gzip compressed
            try:
                data = json.loads(data)
            except Exception as e:
                logger.exception(e)
                logger.error(f'Failed unpacking redis query. key: {key}, response: {data}')
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
        return f"{self.table_name}:{key}:{value}"

    def query(self, index_key, index_value):
        """
        example:
            table = RedisTable('p-bla-bla')
            result_dict = table.query('uuid', '1234')

        :returns response JSON as dict, None if not found, or response string if JSON conversion fails
        """
        key = self.table_index_key(index_key, index_value)
        data = self.get(key)
        decoded_data = self._decode_data(data, key)
        return decoded_data

    def query_multiple(self, index_key, index_values):
        """
        :returns list of response JSONs as dicts, Nones if not found, or response string if JSON conversion fails
        """
        keys = [self.table_index_key(index_key, v) for v in index_values]
        results = self.mget(keys)
        decoded_results = [self._decode_data(r, k) for k, r in zip(keys, results)]
        return decoded_results


class S3FileIO(LogCallsTimeAndOutput):

    buff_size = 1024

    def __init__(self, bucket_name, assume_role=None, log_level=None):
        super().__init__()
        self.assume_role = assume_role
        self.bucket_name = bucket_name
        self.log_level = log_level if log_level is not None else logging.INFO

    def _s3_resource(self):
        creds = {}
        if self.assume_role is not None:
            client = boto3.session.Session().client('sts')
            current_arn = client.get_caller_identity()['Arn']
            if current_arn != self.assume_role:
                assumedRoleObject = client.assume_role(
                    RoleArn=self.assume_role, RoleSessionName="AssumeRoleSession1")
                credentials = assumedRoleObject['Credentials']
                creds = dict(
                    aws_access_key_id=credentials['AccessKeyId'],
                    aws_secret_access_key=credentials['SecretAccessKey'],
                    aws_session_token=credentials['SessionToken'])
        return boto3.session.Session().resource('s3', **creds)

    def _s3_obj(self, path):
        return self._s3_resource().Bucket(self.bucket_name).Object(path)

    @log_errors(message='Failed writing to S3')
    def write_binary(self, data, remote_path, compress=True):
        if compress:
            try:
                # https://stackoverflow.com/questions/33562394/gzip-raised-overflowerror-size-does-not-fit-in-an-unsigned-int
                data = gzip.compress(data, 1)
            except OverflowError:
                pass
        with io.BytesIO(data) as f:
            self._s3_resource().Bucket(self.bucket_name).\
                upload_fileobj(Fileobj=f, Key=remote_path)

    @log_errors(message='Failed reading from S3')
    def read(self, remote_path):
        data = self._s3_obj(remote_path).get()['Body'].read()
        try:
            data = gzip.decompress(data)
        except OSError:
            pass
        return data

    def _stream_to_file(self, srtream, fileobj):
        while fileobj.write(srtream.read(self.buff_size)):
            pass
        fileobj.flush()
        fileobj.seek(0)

    def _stream_obj_to_file(self, remote_path, fileobj):
        # https://stackoverflow.com/questions/7624900/how-can-i-use-boto-to-stream-a-file-out-of-amazon-s3-to-rackspace-cloudfiles
        obj_body = self._s3_obj(remote_path).get()['Body']
        return self._stream_to_file(obj_body, fileobj)

    def _download_through_disk(self, remote_path, local_fileobj):
        with tempfile.NamedTemporaryFile(delete=True) as temp:
            self._stream_obj_to_file(remote_path=remote_path, fileobj=temp)
            try:
                with gzip.open(temp) as gzipfile:
                    with io.BufferedReader(gzipfile) as gzipbuffered:
                        return self._stream_to_file(gzipbuffered, local_fileobj)
            except Exception as e:
                logger.info('_download_through_disk: failed gzip read, assuming regular binary')
                temp.seek(0)
                return self._stream_to_file(temp, local_fileobj)

    def pickle(self, obj, remote_path, compress=True):
        logger.log(self.log_level, f'S3: pickling to {remote_path}')
        return self.write_binary(pickle.dumps(obj), remote_path, compress=compress)

    def unpickle(self, remote_path, in_memory=True):
        logger.log(self.log_level, f'S3: unpickling from {remote_path}')
        if in_memory:
            return pickle.loads(self.read(remote_path))
        else:
            with tempfile.NamedTemporaryFile(delete=True) as temp:
                self._download_through_disk(remote_path, temp)
                return pickle.load(temp)

    def local_to_remote(self, local_path, remote_path, compress=True):
        logger.log(self.log_level, f'S3: copying from {local_path} to {remote_path}')
        with open(local_path, 'rb') as local:
            self.write_binary(local.read(), remote_path, compress=compress)

    def remote_to_local(self, remote_path, local_path, overwrite=True, in_memory=True):
        if not os.path.exists(local_path) or overwrite:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            logger.log(self.log_level, f'S3: copying from {remote_path} to {local_path}')
            with open(local_path, 'wb') as local:
                if in_memory:
                    local.write(self.read(remote_path))
                else:
                    self._download_through_disk(remote_path, local)

    def listdir(self, path):
        s3 = self._s3_resource().Bucket(self.bucket_name)
        return [object_summary.key for object_summary in s3.objects.filter(Prefix=path)]

    def cache_multiple_from_remote(self, paths, destination, overwrite=True):
        cached_paths = []
        for s3_path in paths:
            local_path = os.path.join(destination, s3_path)
            cached_paths.append(local_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            if not os.path.exists(local_path) or overwrite:
                self.remote_to_local(s3_path, local_path)
        return cached_paths


class Emailer:
    def __init__(self, from_email='name@domain.com', backend='SES:us-west-2'):
        self.from_email = from_email
        self.backend = backend

    def _basic_message(self, to, subject='', body=''):
        msg = MIMEMultipart()
        msg['Subject'] = subject
        msg['From'] = self.from_email
        msg['To'] = to
        msg.attach(MIMEText(body))
        return msg

    def _SES_region(self):
        parts = self.backend.split('SES:')
        if len(parts) < 2 or not len(parts[1]):
            raise ValueError('Please pass AWS_REGION as part of backend parameter. e.g. backend="SES:us-west-2')
        return parts[1]

    @log_errors()
    def _send_message(self, msg, to):
        to = [to] if isinstance(to, str) else to

        if self.backend=='SMTP':
            s = smtplib.SMTP('localhost')
            s.sendmail(self.from_email, to, msg.as_string())
            s.quit()

        elif 'SES' in self.backend:
            # https://docs.aws.amazon.com/ses/latest/DeveloperGuide/send-email-raw.html
            client = boto3.session.Session().client('ses', region_name=self._SES_region())
            try:
                client.send_raw_email(
                    Source=self.from_email,
                    Destinations=to,
                    RawMessage={'Data': msg.as_string(),}
                )
            # Display an error if something goes wrong.
            except ClientError as e:
                logger.error(e.response['Error']['Message'])
                raise e
        else:
            raise ValueError(f'Unknown email backend: {self.backend}')

    def _text_attachment(self, text_file):
        with open(text_file) as fp:
            attachment_msg = MIMEText(fp.read())
        attachment_msg.add_header('Content-Disposition', 'attachment', filename=text_file)
        return attachment_msg

    def _image_attachment(self, image_file):
        with open(image_file, 'rb') as fp:
            attachment_msg = MIMEImage(fp.read())
        attachment_msg.add_header('Content-ID', 'attachment', filename=image_file)
        return attachment_msg

    @staticmethod
    def _default_subject(text_files):
        return ','.join([os.path.split(f)[-1] for f in text_files])

    @log_errors()
    def send_simple_message(self, to, subject='', body=''):
        msg = self._basic_message(to, subject=subject, body=body)
        self._send_message(msg, to)

    def _read_text_file(self, text_file):
        with open(text_file, 'rt') as f:
            return f.read()

    @log_errors()
    def send_text_file(self, to, text_file, subject=None, attach=True):
        if subject is None:
            subject = self._default_subject([text_file])

        body = self._read_text_file(text_file)

        if attach:
            return self.send_text_files_attached(
                to=to, text_files=[text_file], subject=subject, body=body)
        else:
            msg = self._basic_message(to, subject=subject, body=body)
            self._send_message(msg, to)

    @log_errors()
    def send_text_files_attached(self, to, text_files, body='', subject=None):
        text_files = [text_files] if isinstance(text_files, str) else text_files

        if os.path.exists(body):
            body = self._read_text_file(body)

        if subject is None:
            subject = self._default_subject(text_files)

        msg = self._basic_message(to, subject=subject, body=body)
        for f in text_files:
            msg.attach(self._text_attachment(f))
        self._send_message(msg, to)

    @log_errors()
    def send_image_attached(self, to, image_file, body='', subject=None):
        if subject is None:
            subject = self._default_subject([image_file])
        msg = self._basic_message(to, subject=subject, body=body)
        msg.attach(self._image_attachment(image_file))
        self._send_message(msg, to)


def hash_str(obj):
    # creates a hash string representing the input object
    obj_str = pickle.dumps(obj)  # pickle is used for speed in case a large object is passed like a DF
    return md5(obj_str).hexdigest()


class DataFrameDiskCacher:
    """
    This is a class that helps cache dataframe files to disk
    """
    def __init__(self,
                 cache_file_pattern='cache_file_%s.tmp',
                 disk_cache_dir=None,
                 salt=None):
        self.cache_file_pattern = cache_file_pattern
        self.disk_cache_dir = disk_cache_dir
        self.salt = salt

    @staticmethod
    def file_age_days(filepath):
        return (time.time() - os.path.getmtime(filepath)) / 86400

    def cache_filepath(self, *args, **kwargs):
        cache_obj = tuple(hash_str(a) for a in args) + \
                    tuple((hash_str(k), hash_str(v)) for k, v in kwargs.items()) + \
                    (self.salt,)
        if self.disk_cache_dir is None:
            logger.warning('Attempting to use cache but cache dir was not defined. Not using cache.')
        else:
            if not os.path.exists(self.disk_cache_dir):
                logger.warning("Cache dir doesn't exist, trying to create.")
                os.makedirs(self.disk_cache_dir, exist_ok=True)
            return os.path.join(self.disk_cache_dir, self.cache_file_pattern % hash_str(cache_obj))

    @classmethod
    def is_cache_valid(cls, cache_path, valid_days=None):
        return cache_path \
               and os.path.exists(cache_path) \
               and (valid_days is None or cls.file_age_days(cache_path) <= valid_days)

    def with_file_cache(self, path_func=None):
        def decorator(func):
            @functools.wraps(func)
            def inner(*args, **kwargs):
                nonlocal self, path_func
                read_from_cache = kwargs.pop('read_from_cache', False)
                save_to_cache = kwargs.pop('save_to_cache', True)
                cache_valid_days = kwargs.pop('cache_valid_days', None)

                if not read_from_cache and not save_to_cache:
                    # short circuit everything if cache not requested
                    return func(*args, **kwargs)

                path_func = path_func or self.cache_filepath
                cache_path = path_func(*args, **kwargs)
                cache_valid = self.is_cache_valid(cache_path, valid_days=cache_valid_days)

                read_cache_attempt = read_from_cache and cache_valid

                # using pickle here because pickling stores the dataframe more reliably
                # (data types and other information may have changed or lost during write/read of csv)

                if read_cache_attempt:
                    # df = pd.read_sv(cache_path, keep_default_na=False, na_values=NA_VALUES)
                    df = pd.read_pickle(cache_path)
                    logger.info(f'Read cache file from {cache_path}')
                else:
                    if read_from_cache:
                        logger.warning(f'Cache file not found/valid, attempting to create ({cache_path})')
                    df = func(*args, **kwargs)

                if save_to_cache and cache_path and not read_cache_attempt:
                    # df.to_csv(cache_path, index=None)
                    df.to_pickle(cache_path)

                return df

            return inner
        return decorator


class DBDFReaderWithCache(LogCallsTimeAndOutput):
    def __init__(self, disk_cache_dir=None, cache_salt=None, **kwargs):
        super().__init__(**kwargs)
        self.disk_query_cacher = DataFrameDiskCacher(
            cache_file_pattern='cached_db_df_%s.pkl',
            disk_cache_dir=disk_cache_dir,
            salt=cache_salt)

    @abstractmethod
    def _fetch_dataframe(self, query: str):
        return pd.DataFrame()

    def fetch_dataframe(self, query: str,
                        save_to_cache=False,
                        read_from_cache=False,
                        cache_valid_days=None):
        cacher = self.disk_query_cacher
        # using the decorator without decorating because
        # the path func only can be defined after initialisations
        fetch_with_cache = cacher.with_file_cache()(self._fetch_dataframe)
        return fetch_with_cache(query=query,
                                save_to_cache=save_to_cache,
                                read_from_cache=read_from_cache,
                                cache_valid_days=cache_valid_days)


class PostgressDFReader(DBDFReaderWithCache):

    def __init__(self,
                 user=None, password=None, database=None, host=None, port=None,
                 disk_cache_dir=None, cache_salt=None, **kwargs):
        super().__init__(disk_cache_dir=disk_cache_dir, cache_salt=cache_salt, **kwargs)
        self.pg_user = user
        self.pg_password = password
        self.pg_database = database
        self.pg_host = host
        self.pg_port = port

    def _connection_params(self):
        return dict(
            user=self.pg_user,
            password=self.pg_password,
            database=self.pg_database,
            host=self.pg_host,
            port=self.pg_port)

    def _fetch_dataframe(self, query: str):
        # this part is with async / await because we're using
        # the asyncpg library (for the speed) and it is ONLY async

        async def run():
            conn = await asyncpg.connect(**self._connection_params())
            stmt = await conn.prepare(query)
            columns = [a.name for a in stmt.get_attributes()]
            data = await stmt.fetch()
            await conn.close()
            return pd.DataFrame(data, columns=columns)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(run())


PostgressReader = PostgressDFReader  # alias for backwards compat


