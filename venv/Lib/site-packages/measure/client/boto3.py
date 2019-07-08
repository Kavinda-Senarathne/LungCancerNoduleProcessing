# -*- coding: utf-8 -*-
from __future__ import absolute_import

# Standard Library
from os import environ

# External Libraries
from measure.client.base import BaseClient


try:
    import boto3
except ImportError:
    boto3_Client = NotImplementedError


class Boto3Client(BaseClient):
    def __init__(
        self,
        aws_access_key_id=None,
        aws_secret_access_key=None,
        region_name=None
    ):
        if not any([aws_access_key_id, aws_secret_access_key]):
            try:
                aws_access_key_id = environ['AWS_ACCESS_KEY_ID']
                aws_secret_access_key = environ['AWS_SECRET_ACCESS_KEY']
            except KeyError:
                raise Exception("You must provide AWS keys either in Env or App")

        region_name = region_name or environ.get('AWS_DEFAULT_REGION', None) or 'us-east-1'

        session = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name
        )
        self.client = session.client('cloudwatch')

    def split_prefix_name(self, prefix_name):
        parts = prefix_name.split('.')
        prefix = parts[:-1]
        name = parts[-1:][0]
        return ".".join(prefix), name

    def submit_metric(self, namespace, metric_name, value, unit='None'):
        self.client.put_metric_data(
            Namespace=namespace,
            MetricData=[
                {
                    'MetricName': metric_name,
                    'Value': value,
                    'Unit': unit
                }
            ]
        )

    def timing(self, prefix_name, value, sample_rate=None):
        namespace, metric_name = self.split_prefix_name(prefix_name)
        self.submit_metric(namespace, metric_name, value, unit='Seconds')

    def update_stats(self, prefix_name, value, sample_rate=None):
        namespace, metric_name = self.split_prefix_name(prefix_name)
        self.submit_metric(namespace, metric_name, value, unit='None')

    def guage(self, prefix_name, value, sample_rate=None):
        namespace, metric_name = self.split_prefix_name(prefix_name)
        self.submit_metric(namespace, metric_name, value, unit='None')

    def send(self, prefix_name, value, sample_rate=None):
        namespace, metric_name = self.split_prefix_name(prefix_name)
        self.submit_metric(namespace, metric_name, value, unit='None')

    def mark(self, prefix_name, value, sample_rate=None):
        namespace, metric_name = self.split_prefix_name(prefix_name)
        self.submit_metric(namespace, metric_name, value, unit='None')
