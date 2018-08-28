import logging
import functools
import os
import time
import inspect
from types import FunctionType
from abc import ABC, ABCMeta
from threading import Thread
from psutil import virtual_memory, cpu_percent

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
        sys_monitor = ResourceMonitor().start()

        start = time.time()

        result = fn(*args, **kwargs)

        elapsed = time.time() - start
        duration_str = '%.2f' % elapsed

        sys_monitor.stop()
        mem_str = 'mem: %s%%(peak:%s%%) cpu:%d%%' % \
                  (sys_monitor.current_memory, sys_monitor.peak_memory, int(sys_monitor.avg_cpu_load))

        ret_str = variable_info(result)

        stack_depth = get_stack_depth()

        fn_str = function_name_with_class(fn)

        msg = ' ' * stack_depth + \
              '%s, elapsed: %s, returned: %s, sys %s' % \
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


class ResourceMonitor:
    def __init__(self, interval=0.2):
        self.interval = interval
        self._init_counters()
        self._thread = None
        self._run_condition = False

    def _init_counters(self):
        self.current_memory = 0
        self.peak_memory = 0
        self.avg_cpu_load = 0
        self._n_measurements = 0

    def __del__(self):
        if self._run_condition:
            self._run_condition = False
            self._thread.join(self.interval + 0.1)

    @staticmethod
    def _current():
        try:
            return virtual_memory().percent, cpu_percent()
        except KeyError:
            # for some reason there's a KeyError: ('psutil',) in psutil
            return 0, 0

    def _measure(self):
        cur_mem, cur_cpu = self._current()
        self.current_memory = cur_mem
        self.peak_memory = max(self.peak_memory, cur_mem)
        self.avg_cpu_load = (self.avg_cpu_load * self._n_measurements + cur_cpu) / \
                            (self._n_measurements + 1)
        self._n_measurements += 1

    def _thread_loop(self):
        while self._run_condition:
            self._measure()
            time.sleep(self.interval)

    def start(self):
        self._init_counters()
        if self._thread is not None:
            self._thread.join(0)
        self._run_condition = True
        self._thread = Thread(target=self._thread_loop, name='ResourceMonitor')
        self._thread.start()
        return self

    def stop(self):
        self._run_condition = False
        self._measure()


def get_stack_depth():
    try:
        return len(inspect.stack(context=0))
    except (IndexError, RuntimeError) as e:
        # there is a bug in inspect module: https://github.com/ipython/ipython/issues/1456/
        # another one: https://bugs.python.org/issue13487
        return 0


def function_name_with_class(fn):
    cls = get_class_that_defined_method(fn)
    cls_str = cls.__name__ + '.' if cls else ''
    return cls_str + fn.__name__


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
                      meth.__qualname__.split('.<locals>', 1)[0].rsplit('.', 1)[0], None)
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


def override_defaults(defaults):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            params = inspect.signature(f).parameters

            new_defaults = [defaults[p] for p in params
                            if params[p].default is not params[p].empty and p in defaults]
            old_defaults = [params[p].default for p in params
                            if params[p].default is not params[p].empty and p in defaults]

            assert f.__defaults__==tuple(old_defaults)

            # Backup original function defaults.
            original_defaults = f.__defaults__

            # Set the new function defaults.
            f.__defaults__ = tuple(new_defaults)

            return_value = f(*args, **kwargs)

            # Restore original defaults (required to keep this trick working.)
            f.__defaults__ = original_defaults

            return return_value

        return wrapper

    return decorator


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

    @staticmethod
    def do_not_decorate(f):
        """
        this is for excluding functions from decorating
        :return: the logging decorator if verbose is True, empty decorator otherwise
        """
        setattr(f, 'decorate', False)
        return f


def log_errors(logger=None, message=None, return_on_error=None):
    def decorator(fn):
        @functools.wraps(fn)
        def inner(*args, **kwargs):
            nonlocal logger, message, return_on_error
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                if logger is None:
                    logger = simple_logger
                logger.exception(e)
                fn_str = function_name_with_class(fn)
                msg_str = ', Message: %s' % message if message else ''
                logger.error('Failed: %s, Error: %s %s' %(fn_str, str(e), msg_str))
                return return_on_error
        return inner
    return decorator


def root_caller_file():
    return os.path.splitext(os.path.split(inspect.stack()[-1].filename)[1])[0]
