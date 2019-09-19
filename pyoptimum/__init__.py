import base64
import warnings
import requests


class Client:

    def __init__(self, **kwargs):

        # username and password
        self.username = kwargs.get('username', None)
        self.password = kwargs.get('password', None)

        # token
        self.token = kwargs.get('token', None)

        if not self.token and (not self.username or not self.password):
            warnings.warn("Token has not been provided and username/password are not valid.")

        # base url
        self.base_url = kwargs.get('base_url', 'https://optimize.vicbee.net/api')
        # make sure base_url does not have trailing /
        while self.base_url[-1] == '/':
            self.base_url = self.base_url[:-1]

        # auto token renewal
        self.auto_token_renewal = kwargs.get('auto_token_renewal', True)

    def get_token(self):

        basic = base64.b64encode(bytes('{}:{}'.format(self.username, self.password), 'utf-8')).decode('utf-8')
        headers = {
            'Content-type': 'application/json',
            'Accept': 'application/json',
            'Authorization': 'Basic ' + basic
        }

        resp = requests.get(self.base_url + '/get_token', headers=headers)

        if resp.ok:

            json = resp.json()
            self.token = json.get('token')

        else:

            resp.raise_for_status()

    def call(self, api, data):

        if self.token is None and not self.auto_token_renewal:
            raise Exception('No token available. Call get_token first')

        elif self.auto_token_renewal:
            # try renewing token
            self.get_token()

        headers = {
            'Content-type': 'application/json',
            'Accept': 'application/json',
            'X-Api-Key': self.token
        }
        resp = requests.post(self.base_url + '/' + api,
                             json=data,
                             headers=headers)

        if resp.ok:

            return resp.json()

        else:

            resp.raise_for_status()
