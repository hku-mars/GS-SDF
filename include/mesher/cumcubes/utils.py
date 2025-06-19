# Copyright (c) OpenMMLab. All rights reserved.
import re
from time import time
from typing import List, Tuple, Union, Sequence

import torch
import numpy as np


def scale_to_bound(scale: Union[float, Sequence]) -> Tuple[List[float]]:
    if isinstance(scale, float):
        lower = [0.0, 0.0, 0.0]
        upper = [scale, scale, scale]
    elif isinstance(scale, (list, tuple, np.ndarray, torch.Tensor)):
        if len(scale) == 3:
            lower = [0.0, 0.0, 0.0]
            upper = [i for i in scale]
        elif len(scale) == 2:
            if isinstance(scale[0], float):
                lower = [scale[0]] * 3
                upper = [scale[1]] * 3
            else:
                assert len(scale[0]) == len(scale[1]) == 3
                lower = [i for i in scale[0]]
                upper = [i for i in scale[1]]
        else:
            raise TypeError()
    else:
        raise TypeError()

    return lower, upper



class TimerError(Exception):

    def __init__(self, message):
        self.message = message
        super(TimerError, self).__init__(message)


class Timer:
    """A flexible Timer class.
    Examples:
        >>> import time
        >>> import mmcv
        >>> with mmcv.Timer():
        >>>     # simulate a code block that will run for 1s
        >>>     time.sleep(1)
        1.000
        >>> with mmcv.Timer(print_tmpl='it takes {:.1f} seconds'):
        >>>     # simulate a code block that will run for 1s
        >>>     time.sleep(1)
        it takes 1.0 seconds
        >>> timer = mmcv.Timer()
        >>> time.sleep(0.5)
        >>> print(timer.since_start())
        0.500
        >>> time.sleep(0.5)
        >>> print(timer.since_last_check())
        0.500
        >>> print(timer.since_start())
        1.000
    """
    def __init__(self, print_tmpl=None, start=True):
        self._is_running = False
        if (print_tmpl
                is not None) and not re.findall(r"({:.*\df})", print_tmpl):
            print_tmpl += " {:.3f}"
            # raise ValueError("`print_tmpl` must has the `{:.nf}` to show time.")
        self.print_tmpl = print_tmpl if print_tmpl else "{:.3f}"
        if start:
            self.start()

    @property
    def is_running(self):
        """bool: indicate whether the timer is running"""
        return self._is_running

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        print(self.print_tmpl.format(self.since_last_check()))
        self._is_running = False

    def start(self):
        """Start the timer."""
        if not self._is_running:
            self._t_start = time()
            self._is_running = True
        self._t_last = time()

    def since_start(self):
        """Total time since the timer is started.
        Returns:
            float: Time in seconds.
        """
        if not self._is_running:
            raise TimerError('timer is not running')
        self._t_last = time()
        return self._t_last - self._t_start

    def since_last_check(self):
        """Time since the last checking.
        Either :func:`since_start` or :func:`since_last_check` is a checking
        operation.
        Returns:
            float: Time in seconds.
        """
        if not self._is_running:
            raise TimerError('timer is not running')
        dur = time() - self._t_last
        self._t_last = time()
        return dur

