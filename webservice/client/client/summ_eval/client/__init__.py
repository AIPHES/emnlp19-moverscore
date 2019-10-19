#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import threading
import time
import uuid
import warnings
from collections import namedtuple
from functools import wraps

import numpy as np
import zmq
from zmq.utils import jsonapi

__all__ = ['__version__', 'EvalClient', 'ConcurrentEvalClient']

# in the future client version must match with server version
__version__ = '1.0.0'

if sys.version_info >= (3, 0):
    from ._py3_var import *
else:
    from ._py2_var import *

_Response = namedtuple('_Response', ['id', 'content'])
Response = namedtuple('Response', ['id', 'scores'])


class EvalClient:
    def __init__(self, ip='localhost', port=5555, port_out=5556,
                 output_fmt='dict', show_server_config=False,
                 identity=None, check_version=True, check_length=True,
                 check_token_info=True, ignore_all_checks=False,
                 timeout=-1):
        """ A client object connected to a EvalServer

        Create a EvalClient that connects to a EvalServer.
        Note, server must be ready at the moment you are calling this function.
        If you are not sure whether the server is ready, then please set `ignore_all_checks=True`

        You can also use it as a context manager:

        .. highlight:: python
        .. code-block:: python

            with EvalClient() as bc:
                bc.encode(...)

            # bc is automatically closed out of the context

        :type timeout: int
        :type check_version: bool
        :type check_length: bool
        :type check_token_info: bool
        :type ignore_all_checks: bool
        :type identity: str
        :type show_server_config: bool
        :type output_fmt: str
        :type port_out: int
        :type port: int
        :type ip: str
        :param ip: the ip address of the server
        :param port: port for pushing data from client to server, must be consistent with the server side config
        :param port_out: port for publishing results from server to client, must be consistent with the server side config
        :param output_fmt: the output format of the sentence encodes, either in numpy array or python List[List[float]] (list/dictionary)
        :param show_server_config: whether to show server configs when first connected
        :param identity: the UUID of this client
        :param timeout: set the timeout (milliseconds) for receive operation on the client, -1 means no timeout and wait until result returns
        """

        self.context = zmq.Context()
        self.sender = self.context.socket(zmq.PUSH)
        self.sender.setsockopt(zmq.LINGER, 0)
        self.identity = identity or str(uuid.uuid4()).encode('ascii')
        self.sender.connect('tcp://%s:%d' % (ip, port))

        self.receiver = self.context.socket(zmq.SUB)
        self.receiver.setsockopt(zmq.LINGER, 0)
        self.receiver.setsockopt(zmq.SUBSCRIBE, self.identity)
        self.receiver.connect('tcp://%s:%d' % (ip, port_out))

        self.request_id = 0
        self.timeout = timeout
        self.pending_request = set()
        self.pending_response = {}

        if output_fmt == 'dict':
            self.formatter = lambda x: x
        else:
            raise AttributeError('"output_fmt" must be "dict"')

        self.output_fmt = output_fmt
        self.port = port
        self.port_out = port_out
        self.ip = ip
        self.length_limit = 0

        if not ignore_all_checks and (check_version or show_server_config):
            s_status = self.server_status

            if check_version and s_status['server_version'] != self.status['client_version']:
                raise AttributeError('version mismatch! server version is %s but client version is %s!\n'
                                     'consider "pip install -U summ-eval-server summ-eval-client"\n'
                                     'or disable version-check by "EvalClient(check_version=False)"' % (
                                         s_status['server_version'], self.status['client_version']))


            if show_server_config:
                self._print_dict(s_status, 'server config:')

    def close(self):
        """
            Gently close all connections of the client. If you are using EvalClient as context manager,
            then this is not necessary.

        """
        self.sender.close()
        self.receiver.close()
        self.context.term()

    def _send(self, msg, msg_len=0):
        self.request_id += 1
        self.sender.send_multipart([self.identity, msg, b'%d' % self.request_id, b'%d' % msg_len])
        self.pending_request.add(self.request_id)
        return self.request_id

    def _recv(self, wait_for_req_id=None):
        try:
            while True:
                # a request has been returned and found in pending_response
                if wait_for_req_id in self.pending_response:
                    response = self.pending_response.pop(wait_for_req_id)
                    return _Response(wait_for_req_id, response)

                # receive a response
                response = self.receiver.recv_multipart()
                request_id = int(response[-1])

                # if not wait for particular response then simply return
                if not wait_for_req_id or (wait_for_req_id == request_id):
                    self.pending_request.remove(request_id)
                    return _Response(request_id, response)
                elif wait_for_req_id != request_id:
                    self.pending_response[request_id] = response
                    # wait for the next response
        except Exception as e:
            raise e
        finally:
            if wait_for_req_id in self.pending_request:
                self.pending_request.remove(wait_for_req_id)

    def _recv_scores(self, wait_for_req_id=None):
        request_id, response = self._recv(wait_for_req_id)
        scores = jsonapi.loads(response[1])
        return Response(request_id, scores)

    @property
    def status(self):
        """
            Get the status of this EvalClient instance

        :rtype: dict[str, str]
        :return: a dictionary contains the status of this EvalClient instance

        """
        return {
            'identity': self.identity,
            'num_request': self.request_id,
            'num_pending_request': len(self.pending_request),
            'pending_request': self.pending_request,
            'output_fmt': self.output_fmt,
            'port': self.port,
            'port_out': self.port_out,
            'server_ip': self.ip,
            'client_version': __version__,
            'timeout': self.timeout
        }

    def _timeout(func):
        @wraps(func)
        def arg_wrapper(self, *args, **kwargs):
            if 'blocking' in kwargs and not kwargs['blocking']:
                # override client timeout setting if `func` is called in non-blocking way
                self.receiver.setsockopt(zmq.RCVTIMEO, -1)
            else:
                self.receiver.setsockopt(zmq.RCVTIMEO, self.timeout)
            try:
                return func(self, *args, **kwargs)
            except zmq.error.Again as _e:
                t_e = TimeoutError(
                    'no response from the server (with "timeout"=%d ms), please check the following:'
                    'is the server still online? is the network broken? are "port" and "port_out" correct? '
                    'are you encoding a huge amount of data whereas the timeout is too small for that?' % self.timeout)
                if _py2:
                    raise t_e
                else:
                    _raise(t_e, _e)
            finally:
                self.receiver.setsockopt(zmq.RCVTIMEO, -1)

        return arg_wrapper

    @property
    @_timeout
    def server_status(self):
        """
            Get the current status of the server connected to this client

        :return: a dictionary contains the current status of the server connected to this client
        :rtype: dict[str, str]

        """
        req_id = self._send(b'SHOW_CONFIG')
        return jsonapi.loads(self._recv(req_id).content[1])

    @_timeout
    def eval(self, pairs):
        """ Encode a list of strings to a list of vectors

        `texts` should be a list of strings, each of which represents a sentence.
        If `is_tokenized` is set to True, then `texts` should be list[list[str]],
        outer list represents sentence and inner list represent tokens in the sentence.
        Note that if `blocking` is set to False, then you need to fetch the result manually afterwards.

        .. highlight:: python
        .. code-block:: python

            with EvalClient() as bc:
                # evaluate pair of summary and references untokenized sentences
                bc.eval([['summary'], [ref]])
            :rtype: dictionary {}

        """
        
        req_id = self._send(jsonapi.dumps(pairs), len(pairs))
        
        r = self._recv_scores(req_id)
        
        return r.scores

    def fetch(self, delay=.0):
        """ Fetch the encoded vectors from server, use it with `encode(blocking=False)`

        Use it after `encode(texts, blocking=False)`. If there is no pending requests, will return None.
        Note that `fetch()` does not preserve the order of the requests! Say you have two non-blocking requests,
        R1 and R2, where R1 with 256 samples, R2 with 1 samples. It could be that R2 returns first.

        To fetch all results in the original sending order, please use `fetch_all(sort=True)`

        :type delay: float
        :param delay: delay in seconds and then run fetcher
        :return: a generator that yields request id and encoded vector in a tuple, where the request id can be used to determine the order
        :rtype: Iterator[tuple(int, numpy.ndarray)]

        """
        time.sleep(delay)
        while self.pending_request:
            yield self._recv_scores()

    def fetch_all(self, sort=True, concat=False):
        """ Fetch all encoded vectors from server, use it with `encode(blocking=False)`

        Use it `encode(texts, blocking=False)`. If there is no pending requests, it will return None.

        :type sort: bool
        :type concat: bool
        :param sort: sort results by their request ids. It should be True if you want to preserve the sending order
        :param concat: concatenate all results into one ndarray
        :return: encoded sentence/token-level embeddings in sending order
        :rtype: numpy.ndarray or list[list[float]]

        """
        if self.pending_request:
            tmp = list(self.fetch())
            if sort:
                tmp = sorted(tmp, key=lambda v: v.id)
            tmp = [v.content for v in tmp]
            if concat:
                if self.output_fmt == 'dict':
                    tmp = [vv for v in tmp for vv in v]
            return tmp

    @staticmethod
    def _check_input_lst_str(pairs):
        if not isinstance(pairs, list):
            raise TypeError('"%s" must be %s, but received %s' % (pairs, type([]), type(pairs)))
        if not len(pairs):
            raise ValueError(
                '"%s" must be a non-empty list, but received %s with %d elements' % (pairs, type(pairs), len(pairs)))
        for idx, s in enumerate(pairs):
            if not isinstance(s, _str):
                raise TypeError('all elements in the list must be %s, but element %d is %s' % (type(''), idx, type(s)))
            if not s.strip():
                raise ValueError(
                    'all elements in the list must be non-empty string, but element %d is %s' % (idx, repr(s)))
            if _py2:
                pairs[idx] = _unicode(pairs[idx])

    @staticmethod
    def _check_input_lst_lst_str(pairs):
        if not isinstance(pairs, list):
            raise TypeError('"pairs" must be %s, but received %s' % (type([]), type(pairs)))
        if not len(pairs):
            raise ValueError(
                '"pairs" must be a non-empty list, but received %s with %d elements' % (type(pairs), len(pairs)))
        for s in pairs:
            EvalClient._check_input_lst_str(s)

    @staticmethod
    def _print_dict(x, title=None):
        if title:
            print(title)
        for k, v in x.items():
            print('%30s\t=\t%-30s' % (k, v))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class ConcurrentEvalClient(EvalClient):
    def __init__(self, max_concurrency=10, **kwargs):
        """ A thread-safe client object connected to a EvalServer

        Create a EvalClient that connects to a EvalServer.
        Note, server must be ready at the moment you are calling this function.
        If you are not sure whether the server is ready, then please set `check_version=False` and `check_length=False`

        :type max_concurrency: int
        :param max_concurrency: the maximum number of concurrent connections allowed

        """
        try:
            from summ_eval.client import EvalClient
        except ImportError:
            raise ImportError('EvalClient module is not available, it is required for serving HTTP requests.'
                              'Please use "pip install -U summ-eval-client" to install it.'
                              'If you do not want to use it as an HTTP server, '
                              'then remove "-http_port" from the command line.')

        self.available_bc = [EvalClient(**kwargs) for _ in range(max_concurrency)]
        self.max_concurrency = max_concurrency

    def close(self):
        for bc in self.available_bc:
            bc.close()

    def _concurrent(func):
        @wraps(func)
        def arg_wrapper(self, *args, **kwargs):
            try:
                bc = self.available_bc.pop()
                f = getattr(bc, func.__name__)
                r = f if isinstance(f, dict) else f(*args, **kwargs)
                self.available_bc.append(bc)
                return r
            except IndexError:
                raise RuntimeError('Too many concurrent connections!'
                                   'Try to increase the value of "max_concurrency", '
                                   'currently =%d' % self.max_concurrency)

        return arg_wrapper

    @_concurrent
    def encode(self, **kwargs):
        pass

    @property
    @_concurrent
    def server_status(self):
        pass

    @property
    @_concurrent
    def status(self):
        pass

    def fetch(self, **kwargs):
        raise NotImplementedError('Async encoding of "ConcurrentEvalClient" is not implemented yet')

    def fetch_all(self, **kwargs):
        raise NotImplementedError('Async encoding of "ConcurrentEvalClient" is not implemented yet')

    def encode_async(self, **kwargs):
        raise NotImplementedError('Async encoding of "ConcurrentEvalClient" is not implemented yet')
