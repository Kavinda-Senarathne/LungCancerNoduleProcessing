# -*- coding: utf-8 -*-

from __future__ import absolute_import

from .stat import (
    Stat,
    StatDict,
)


class Gauge(Stat):
    """
    A discrete number, i.e. not a rate.
    """
    _function = 'gauge'
    _alias = 'set'

    def set(self, n):
        self.apply(n)


class GaugeDict(StatDict):
    _stat_class = Gauge
