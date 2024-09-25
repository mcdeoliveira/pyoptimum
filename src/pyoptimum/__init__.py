import base64
import re
from json import JSONDecodeError
from typing import Optional, Any

import json


class PyOptimumException(Exception):
    """
    pyoptimum Exception
    """
    pass


class Client:
    """
    Client object to facilitate connection to the
    `optimize.vicbee.net <https:optimize.vicbee.net>`_ optimization api

    Calls will be made to a URL of the form:

    ``base_url/api/prefix/entry_point``

    in which ``entry_point`` is set in :meth:`pyoptimum.Client.call`.

    :param username: the username
    :param password: the password
    :param token: an authentication token
    :param auto_token_renewal: whether to automatically renew an expired token
    :param base_url: the api base url
    :param api: the target api
    :param prefix: the target api prefix
    :param auth_url: the api auth url (optional)
    """

    SLASH_END = re.compile('/+$')
    SLASH_START = re.compile('^/+')

    def __init__(self,
                 username: Optional[str] = None, password: Optional[str] = None,
                 token: Optional[str] = None,
                 auto_token_renewal: bool = True,
                 base_url: str = 'https://optimize.vicbee.net',
                 api: str = 'optimize',
                 prefix: str = 'api',
                 auth_url: str = ''):

        # username and password
        self.username = username
        self.password = password

        # token
        self.token = token

        # auto token renewal
        self.auto_token_renewal = auto_token_renewal

        if not self.token and (not self.username or not self.password):
            raise PyOptimumException("Token or username/password "
                                     "have not been provided.")

        # base url
        self.base_url = Client.url_join(base_url, api, prefix)

        # auth url
        self.auth_url = auth_url or self.base_url

        # initialize detail
        self.detail = None

    @staticmethod
    def url_join(*args: str) -> str:
        """
        Sanitizes and join url parts

        The main role of this function is to remove slashes at the beginning or end of url parts

        :param args: the parts of the url
        :return: the sanitized url
        """
        return '/'.join([x for x in
                         [Client.SLASH_START.sub('',
                                                 Client.SLASH_END.sub('', y))
                          for y in args] if x])

    def get_token(self) -> None:
        """
        Retrieve authentication token
        """
        from requests import get

        basic = base64.b64encode(bytes('{}:{}'.format(self.username, self.password),
                                       'utf-8')).decode('utf-8')
        headers = {
            'Content-type': 'application/json',
            'Accept': 'application/json',
            'Authorization': 'Basic ' + basic
        }

        response = get(Client.url_join(self.auth_url, 'get_token'), headers=headers)
        self.detail = None

        if response.ok:
            try:
                self.token = response.json().get('token')
            except (KeyError, JSONDecodeError) as e:
                raise PyOptimumException(f"Error while retrieving token: Invalid token: {e}")
            except Exception as e:
                raise PyOptimumException(f"Error while retrieving token: {e}")
        else:
            response.raise_for_status()

    def call(self, entry_point: str, data: Any) -> Any:
        """
        Calls the api ``entry_point`` with ``data``

        :param entry_point: the api entry point
        :param data: the data
        :return: dictionary with the response
        """

        from requests import post

        if self.token is None and not self.auto_token_renewal:
            raise PyOptimumException('No token available. Call get_token first')

        elif self.auto_token_renewal:
            # try renewing token
            self.get_token()

        # See https://github.com/psf/requests/issues/6014
        headers = {
            'Content-type': 'application/json',
            'Accept': 'application/json',
            'X-Api-Key': self.token
        }
        response = post(Client.url_join(self.base_url, entry_point),
                        data=json.dumps(data),
                        headers=headers)

        if response.ok:

            try:
                self.detail = None
                return response.json()
            except JSONDecodeError as e:
                raise PyOptimumException(f"Invalid response: {e}")

        else:

            if response.status_code == 400:
                content = json.loads(response.content)
                self.detail = content.get('detail', None)
                if self.detail:
                    raise PyOptimumException(self.detail)
            response.raise_for_status()

class AsyncClient(Client):
    """
    Async client object to facilitate connection to the
    `optimize.vicbee.net <https:optimize.vicbee.net>`_ optimization api

    Calls will be made to a URL of the form:

    ``base_url/api/prefix/entry_point``

    in which ``entry_point`` is set in :meth:`pyoptimum.Client.call`.

    :param username: the username
    :param password: the password
    :param token: an authentication token
    :param auto_token_renewal: whether to automatically renew an expired token
    :param base_url: the api base url
    :param api: the target api
    :param prefix: the target api prefix
    :param auth_url: the api auth url (optional)
    """

    async def get_token(self) -> None:
        """
        Retrieve authentication token
        """
        from aiohttp import ClientSession

        basic = base64.b64encode(bytes('{}:{}'.format(self.username, self.password),
                                       'utf-8')).decode('utf-8')
        headers = {
            'Content-type': 'application/json',
            'Accept': 'application/json',
            'Authorization': 'Basic ' + basic
        }

        self.detail = None
        async with ClientSession() as session:
            async with session.get(Client.url_join(self.auth_url, 'get_token'),
                                   headers=headers, raise_for_status=True) as resp:
                if resp.status < 400:
                    try:
                        response = await resp.json()
                        self.token = response.get('token')
                    except JSONDecodeError as e:
                        raise PyOptimumException(f"Invalid response: {e}")
                else:
                    resp.raise_for_status()


    async def call(self, entry_point: str, data: Any) -> Any:
        """
        Calls the api ``entry_point`` with ``data``

        :param entry_point: the api entry point
        :param data: the data
        :return: dictionary with the response
        """

        from aiohttp import ClientSession

        if self.token is None and not self.auto_token_renewal:
            raise PyOptimumException('No token available. Call get_token first')

        elif self.auto_token_renewal:
            # try renewing token
            await self.get_token()

        # See https://github.com/psf/requests/issues/6014
        headers = {
            'Content-type': 'application/json',
            'Accept': 'application/json',
            'X-Api-Key': self.token
        }
        async with ClientSession() as session:
            async with session.post(Client.url_join(self.base_url, entry_point),
                                    data=json.dumps(data),
                                    headers=headers) as resp:
                if resp.status < 400:

                    try:
                        self.detail = None
                        return await resp.json()
                    except JSONDecodeError as e:
                        raise PyOptimumException(f"Invalid response: {e}")

                elif resp.status == 400:
                    content = json.loads(await resp.content.read())
                    self.detail = content.get('detail', None)
                    if self.detail:
                        raise PyOptimumException(self.detail)

                resp.raise_for_status()
