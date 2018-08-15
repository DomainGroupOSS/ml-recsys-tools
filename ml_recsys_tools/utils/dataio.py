import gzip
import io
import json
import os
import pickle
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import boto3
import redis

from ml_recsys_tools.utils.instrumentation import log_errors
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

    @log_errors(message='Failed writing to S3')
    def write_binary(self, data, remote_path, compress=True):
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

    @log_errors(message='Failed reading from S3')
    def read(self, remote_path):
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


class Emailer:
    def __init__(self, from_email='emailer@reporting'):
        self.from_email = from_email

    def _basic_message(self, to, subject='', body=''):
        msg = MIMEMultipart()
        msg['Subject'] = subject
        msg['From'] = self.from_email
        msg['To'] = to
        msg.attach(MIMEText(body))
        return msg

    def _send_message(self, msg, to):
        s = smtplib.SMTP('localhost')
        to = [to] if isinstance(to, str) else to
        s.sendmail(self.from_email, to, msg.as_string())
        s.quit()

    def _text_attachment(self, text_file):
        with open(text_file) as fp:
            attachment_msg = MIMEText(fp.read())
        attachment_msg.add_header('Content-Disposition', 'attachment', filename=text_file)
        return attachment_msg

    @staticmethod
    def _default_subject(file):
        return os.path.split(file)[-1]

    @log_errors()
    def send_text_file(self, to, text_file, subject=None, attach=True):
        if subject is None:
            subject = self._default_subject(text_file)

        with open(text_file, 'rt') as f:
            body = f.read()

        if attach:
            return self.send_text_file_attached(
                to=to, text_file=text_file, subject=subject, body=body)
        else:
            msg = self._basic_message(to, subject=subject, body=body)
            self._send_message(msg, to)

    @log_errors()
    def send_text_file_attached(self, to, text_file, body='', subject=None):
        if subject is None:
            subject = self._default_subject(text_file)
        msg = self._basic_message(to, subject=subject, body=body)
        msg.attach(self._text_attachment(text_file))
        self._send_message(msg, to)
