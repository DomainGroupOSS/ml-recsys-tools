import base64
import requests
import time


def singleton(cls):
    instances = {}

    def wrapper(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return wrapper


@singleton
class AuthSettings:
    def __init__(self, auth_uri, client_id, secret, scopes):
        self.auth_service_url = auth_uri
        self.client_id = client_id
        self.secret = secret
        self.scopes = scopes


@singleton
class Request:
    """
    """

    def __init__(self, auth_settings, cache):
        self.read_token = ''
        self.auth_service_url = auth_settings.auth_service_url
        self.client_id = auth_settings.client_id
        self.secret = auth_settings.secret
        self.scopes = auth_settings.scopes
        self.cache = cache

    def _load_read_token(self):
        authorization = 'Basic ' + \
                        base64.b64encode(('%s:%s' %
                                          (self.client_id, self.secret)).encode()).decode()
        response = requests.post(self.auth_service_url + '/connect/token',
                                 headers={'Authorization': authorization},
                                 data={'scope': ' '.join(self.scopes),
                                       'grant_type': 'client_credentials'})
        if response.ok:
            oauth_response = response.json()
            access_token = oauth_response['access_token']
            token_type = oauth_response['token_type']
            read_token = token_type + ' ' + access_token
            self._set_cached(self.client_id, read_token, oauth_response['expires_in'] - 60)
        else:
            raise RequestFailure('Failed to retrieve OAuth token.', response=response)
        self.read_token = read_token
        return read_token

    def __call__(self, func):
        def wrapped_func(*args, **kwargs):
            # print(self.read_token)
            self.read_token = self._get_cache(self.client_id)
            if not self.read_token:
                self._load_read_token()
            kwargs = kwargs or {}
            kwargs['headers'] = kwargs.get('headers', {})
            kwargs['headers'].update({'Authorization': self.read_token})
            return func(*args, **kwargs)

        return wrapped_func

    def _get_cache(self, client_id):
        return self.cache.get(client_id)

    def _set_cached(self, client_id, token, expiry):
        return self.cache.set(client_id, token, expiry)


@singleton
class DictCache:
    def __init__(self):
        self.dict = {}

    def not_expired(self, cached_dict):
        return time.time() < cached_dict['expiry']

    def get(self, key, default=None):
        cached = self.dict.get(key, default)
        if cached is not default and self.not_expired(cached):
            return cached['value']
        else:
            return default

    def set(self, key, value, expiry=300):
        self.dict[key] = {'value': value,
                          'expiry': time.time() + expiry}


class RequestFailure(RuntimeError):
    def __init__(self, message, response):
        super(RequestFailure, self).__init__(message)
        self.response = response

# #usage
# CLIENT_ID = 'client-id'
# SECRET = 'secret'
# SCOPES = ['scope1', 'scope1']
# AUTH_SERVICE_URL = 'https://url.com'
#
# settings = AuthSettings(AUTH_SERVICE_URL, CLIENT_ID, SECRET, SCOPES)
# mem_cache = DictCache()
# api_request = Request(settings, mem_cache)
#
# @api_request
# def oauth_get(url, **kwargs):
#     # no 401 403 handler here
#     response = requests.get(url, **kwargs)
#     # print(response)
#     return response
#
# @api_request
# def oauth_post(url, **kwargs):
#     # no 401 403 handler here
#     return requests.post(url, **kwargs)
