# -*- coding: utf-8 -*-

from __future__ import absolute_import

# Standard Library
import logging

from .counter import (
    Counter,
    CounterDict,
)
from .gauge import (
    Gauge,
    GaugeDict,
)
from .meter import (
    Meter,
    MeterDict,
)
from .stat import (
    Stat,
    StatDict,
    Stats,
)
from .timer import (
    Timer,
    TimerDict,
)

logger = logging.getLogger(__name__)


class FakeStat(Timer, Meter, Counter, Gauge, TimerDict, CounterDict, GaugeDict):
    # Meter must precede Counter for the MRO to resolve

    def apply(self, *args, **kwargs):
        logger.error('stat <%s> does not exist', self.name)

    def decrement(self, *args, **kwargs):
        """
        override the meter decrement.
        """
        self.apply(*args, **kwargs)


FakeStatDict = FakeStat
