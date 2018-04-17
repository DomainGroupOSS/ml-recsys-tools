import logging
import functools
import time
import inspect
from types import FunctionType
from abc import ABC, ABCMeta
from threading import Thread
from psutil import virtual_memory

from ml_recsys_tools.utils.logger import simple_logger


class LoggingVerbosity:
    def __init__(self, verbose=True, min_time=1):
        """
        :param verbose: true or false - whether to print to logging.INFO
        :param min_time: minimum time for which to print, if function call is shorter - nothing is printed
        """
        self.verbose = verbose
        self.min_time = min_time

    @property
    def level(self):
        return logging.INFO if self.verbose else logging.DEBUG


LOGGING_VERBOSITY = LoggingVerbosity()


def variable_info(result):
    if hasattr(result, 'shape'):
        shape_str = 'shape: %s' % str(result.shape)
    elif isinstance(result, tuple) and len(result) <= 3:
        shape_str = 'tuple: (' + ','.join([variable_info(el) for el in result]) + ')'
    elif hasattr(result, '__len__'):
        shape_str = 'len: %s' % str(len(result))
    else:
        shape_str = str(result)[:50] + '...'

    ret_str = str(type(result)) + ', ' + shape_str
    return ret_str


def log_time_and_shape(fn):
    @functools.wraps(fn)
    def inner(*args, **kwargs):
        mem_monitor = MaxMemoryMonitor().start()

        start = time.time()

        result = fn(*args, **kwargs)

        elapsed = time.time() - start
        duration_str = '%.2f' % elapsed

        cur_mem, peak_mem = mem_monitor.stop()

        mem_str = '%s%%(peak:%s%%)' % (cur_mem, peak_mem)

        ret_str = variable_info(result)

        stack_depth = get_stack_depth()

        fn_str = class_name(fn) + fn.__name__

        msg = ' ' * stack_depth + \
              '%s, elapsed: %s, returned: %s, sys mem: %s' % \
              (fn_str, duration_str, ret_str, mem_str)

        if elapsed >= LOGGING_VERBOSITY.min_time:
            simple_logger.log(LOGGING_VERBOSITY.level, msg)

        return result

    return inner


def timer_deco(fn):
    log_format = 'function %s: %s s'

    @functools.wraps(fn)
    def inner(*args, **kwargs):
        start = time.time()
        result = fn(*args, **kwargs)
        duration = time.time() - start
        simple_logger.log(LOGGING_VERBOSITY.level, log_format,
                          fn.__name__, duration)
        return result

    return inner


class MaxMemoryMonitor:
    def __init__(self, interval=0.2):
        self.interval = interval
        self.peak_memory = None
        self.thread = None
        self._run_condition = False

    def __del__(self):
        if self._run_condition:
            self._run_condition = False
            self.thread.join(self.interval + 0.1)

    @staticmethod
    def _current():
        try:
            return virtual_memory().percent
        except KeyError:
            # for some reason there's a KeyError: ('psutil',) in psutil
            return 0

    def _measure_peak(self):
        self.peak_memory = max(self.peak_memory, self._current())

    def _thread_loop(self):
        while self._run_condition:
            self._measure_peak()
            time.sleep(self.interval)

    def start(self):
        self._run_condition = True
        self.peak_memory = 0
        self.thread = Thread(target=self._thread_loop, name='MaxMemoryMonitor')
        self.thread.start()
        return self

    def stop(self):
        self._run_condition = False
        self._measure_peak()
        return self._current(), self.peak_memory


def get_stack_depth():
    try:
        return len(inspect.stack())
    except IndexError as e:
        # there is a bug in inspect module: https://github.com/ipython/ipython/issues/1456/
        return 0


def class_name(fn):
    cls = get_class_that_defined_method(fn)
    cls_str = cls.__name__ + '.' if cls else ''
    return cls_str


def get_class_that_defined_method(meth):
    # from https://stackoverflow.com/questions/3589311/
    # get-defining-class-of-unbound-method-object-in-python-3/25959545#25959545
    # modified to return first parent in reverse MRO
    if inspect.ismethod(meth):
        for cls in inspect.getmro(meth.__self__.__class__)[::-1]:
            if cls.__dict__.get(meth.__name__) is meth:
                return cls
        meth = meth.__func__  # fallback to __qualname__ parsing
    if inspect.isfunction(meth):
        cls = getattr(inspect.getmodule(meth),
                      meth.__qualname__.split('.<locals>', 1)[0].rsplit('.', 1)[0])
        if isinstance(cls, type):
            return cls
    return getattr(meth, '__objclass__', None)  # handle special descriptor objects


def collect_named_init_params(cls, skip_empty=True, ignore_classes=(object, ABC)):
    """
    a method to get all named params from all classed in this class' MRO
    can be used to infer the possible search space for hyperparam optimization

    :param skip_empty: whether to skip classes with no named parameters
    :return: a nested dict of {class-name: dict of {named-init-params: default values}}
    """
    params = {}
    for c in inspect.getmro(cls):
        if c not in ignore_classes:
            named_params = [
                p for p in inspect.signature(c.__init__).parameters.values()
                if p.name != 'self'
                   and p.kind != p.VAR_KEYWORD
                   and p.kind != p.VAR_POSITIONAL]

            if skip_empty and not named_params:
                continue

            params[c.__name__] = {
                p.name: p.default
                if p.default is not p.empty else ''
                for p in named_params}
    return params


# https://stackoverflow.com/questions/10067262/automatically-decorating-every-instance-method-in-a-class
# decorate all instance methods (unless excluded) with the same decorator
def decorate_all_metaclass(decorator):
    # check if an object should be decorated
    def do_decorate(attr, value):
        return ('__' not in attr and
                isinstance(value, (FunctionType, classmethod)) and
                getattr(value, 'decorate', True))

    class DecorateAll(ABCMeta):
        def __new__(cls, name, bases, dct):
            if dct.get('decorate', True):
                for attr, value in dct.items():
                    if do_decorate(attr, value):
                        if isinstance(value, classmethod):
                            dct[attr] = classmethod(decorator(value.__func__))
                        else:
                            dct[attr] = decorator(value)
            return super(DecorateAll, cls).__new__(cls, name, bases, dct)

        def __setattr__(self, attr, value):
            if do_decorate(attr, value):
                value = decorator(value)
            super(DecorateAll, self).__setattr__(attr, value)

    return DecorateAll


class LogCallsTimeAndOutput(metaclass=decorate_all_metaclass(log_time_and_shape)):

    def __init__(self, verbose, **kwargs):
        self.verbose = verbose

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, verbose):
        self._verbose = verbose
        if not self.verbose:
            self.decorate = False

    @property
    def logging_decorator(self):
        """
        this is for decorating inner scope functions
        :return: the logging decorator if verbose is True, empty decorator otherwise
        """
        if self.verbose:
            return log_time_and_shape
        else:
            return lambda f: f

