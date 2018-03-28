import json
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