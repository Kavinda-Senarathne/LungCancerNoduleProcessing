# -*- coding: utf-8 -*-
from __future__ import absolute_import

# Standard Library
from contextlib import contextmanager
from functools import wraps
from time import time

from .stat import (
    Stat,
    StatDict,
)


class Timer(Stat):
    """
    Time based stat that is usable via direct call, decorator, or context manager
    """
    _function = 'timing'
    _alias = 'time'

    def time(self, *args):
        """
        Time a function as a decorator or a block as a context manager.
            >>> stat = Timer('foo_latency', 'times latency of foo')

            >>> start = time.time()
            >>> # do work
            >>> stat.time(time.time() - start)

            >>> @stat.time
            >>> def foo():
            >>>     # do work
            >>>     pass

            >>> with stat.time():
            >>>     # do work
            >>>     pass

        """
        if len(args) == 1:
            arg = args[0]
            if callable(arg):
                return self.time_decorator(arg)
            else:
                self.apply(arg)
        else:
            return self.time_contextmanager()

    def time_decorator(self, f):
        """
        Allows for the following syntax:

            >>> @stat.time
            >>> def foo():
            >>>     pass

        """

        @wraps(f)
        def decorator(*args, **kwargs):
            with self.time():
                return f(*args, **kwargs)

        return decorator

    @contextmanager
    def time_contextmanager(self):
        """
        Allows for the following syntax:

            >>> with stat.time():
            >>>     pass

        """
        start = time()
        yield
        end = time()
        self.apply(end - start)


class TimerDict(StatDict):
    _stat_class = Timer
