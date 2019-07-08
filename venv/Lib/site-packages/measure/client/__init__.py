# -*- coding: utf-8 -*-

from __future__ import absolute_import


from .boto3 import Boto3Client
from .pystatsd import PyStatsdClient
from .test import TestStatsdClient

# remove clients that don't exist
for name, client in globals().items():
    if isinstance(client, type) and issubclass(client, NotImplementedError):
        del name
