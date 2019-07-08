# -*- coding: utf-8 -*-
from __future__ import absolute_import

from .counter import Counter
from .stat import StatDict


class Meter(Counter):
    """
    A positive counter that directly represents a rate.
    """

    def mark(self, n=1):
        """
        stat.mark()
        """
        self.increment(n)

    def decrement(self, n=None):
        raise NotImplementedError("Meters do not have the ability to decrement.")


class MeterDict(StatDict):
    _stat_class = Meter
