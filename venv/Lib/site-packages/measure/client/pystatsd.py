# -*- coding: utf-8 -*-

from __future__ import absolute_import

# External Libraries
from measure.client.base import BaseClient

try:
    from pystatsd import Client as pystatsd_Client
except ImportError:
    pystatsd_Client = NotImplementedError


class PyStatsdClient(BaseClient):

    def __init__(self, host='localhost', port=8125, prefix=None):
        self.client = pystatsd_Client(host, port, prefix)

    def timing(self, *args, **kwargs):
        self.client.timing(*args, **kwargs)

    def update_stats(self, *args, **kwargs):
        self.client.update_stats(*args, **kwargs)

    def gauge(self, *args, **kwargs):
        self.client.gauge(*args, **kwargs)

    def send(self, *args, **kwargs):
        self.client.send(*args, **kwargs)
