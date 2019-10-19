import multiprocessing
import os
import random
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime
from itertools import chain
from multiprocessing import Process
from multiprocessing.pool import Pool

import zmq
import zmq.decorators as zmqd
import numpy as np
from termcolor import colored
from zmq.utils import jsonapi

from .helper import *
from .http import EvalHTTPProxy
from .zmq_decor import multi_socket
from metrics.ROUGE import rouge_n
from metrics.MoverScore import sent_mover_score
from metrics.MoverScore import word_mover_score


__all__ = ['__version__', 'EvalServer']
__version__ = '1.0.0'


class ServerCmd:
    terminate = b'TERMINATION'
    show_config = b'SHOW_CONFIG'
    new_job = b'REGISTER'
    data_pair = b'PAIR'
    data_score = b'SCORE'

    @staticmethod
    def is_valid(cmd):
        return any(not k.startswith('__') and v == cmd for k, v in vars(ServerCmd).items())


class EvalServer(threading.Thread):
    def __init__(self, args):
        super().__init__()
        self.logger = set_logger(colored('VENTILATOR', 'magenta'), args.verbose)

        self.data_dir = args.data_dir
        self.num_worker = args.num_worker
        self.num_concurrent_socket = max(8, args.num_worker * 2)  # optimize concurrency for multi-clients
        self.port = args.port
        self.args = args
        self.status_args = {k: (v if k != 'pooling_strategy' else v.value) for k, v in sorted(vars(args).items())}
        self.status_static = {
            'python_version': sys.version,
            'server_version': __version__,
            'pyzmq_version': zmq.pyzmq_version(),
            'zmq_version': zmq.zmq_version(),
            'server_start_time': str(datetime.now()),
        }
        self.processes = []

    def close(self):
        self.logger.info('shutting down...')
        self._send_close_signal()
        for p in self.processes:
            p.close()
        self.join()

    @zmqd.context()
    @zmqd.socket(zmq.PUSH)
    def _send_close_signal(self, _, frontend):
        frontend.connect('tcp://localhost:%d' % self.port)
        frontend.send_multipart([b'', ServerCmd.terminate, b'', b''])

    def run(self):
        self._run()

    @zmqd.context()
    @zmqd.socket(zmq.PULL)
    @zmqd.socket(zmq.PAIR)
    @multi_socket(zmq.PUSH, num_socket='num_concurrent_socket')
    def _run(self, _, frontend, sink, *backend_socks):

        def push_new_job(_job_id, _json_msg, _msg_len):
            # backend_socks[0] is always at the highest priority
            _sock = backend_socks[0]
            _sock.send_multipart([_job_id, _json_msg])

        # bind all sockets
        self.logger.info('bind all sockets')
        frontend.bind('tcp://*:%d' % self.port)
        addr_front2sink = auto_bind(sink)
        addr_backend_list = [auto_bind(b) for b in backend_socks]
        self.logger.info('open %d ventilator-worker sockets' % len(addr_backend_list))

        # start the sink process
        self.logger.info('start the sink')
        proc_sink = EvalSink(self.args, addr_front2sink)
        self.processes.append(proc_sink)
        proc_sink.start()
        addr_sink = sink.recv().decode('ascii')

        # start the backend processes
        process = EvalWorker(1, self.args, addr_backend_list, addr_sink)
        self.processes.append(process)
        process.start()

        # start the http-service process
        if self.args.http_port:
            self.logger.info('start http proxy')
            proc_proxy = EvalHTTPProxy(self.args)
            self.processes.append(proc_proxy)
            proc_proxy.start()

        rand_backend_socket = None
        server_status = ServerStatistic()
        while True:
            try:
                request = frontend.recv_multipart()
                client, msg, req_id, msg_len = request
            except ValueError:
                self.logger.error('received a wrongly-formatted request (expected 4 frames, got %d)' % len(request))
                self.logger.error('\n'.join('field %d: %s' % (idx, k) for idx, k in enumerate(request)), exc_info=True)
            else:
                server_status.update(request)
                if msg == ServerCmd.terminate:
                    break
                elif msg == ServerCmd.show_config:
                    self.logger.info('new config request\treq id: %d\tclient: %s' % (int(req_id), client))
                    status_runtime = {'client': client.decode('ascii'),
                                      'num_process': len(self.processes),
                                      'ventilator -> worker': addr_backend_list,
                                      'worker -> sink': addr_sink,
                                      'ventilator <-> sink': addr_front2sink,
                                      'server_current_time': str(datetime.now()),
                                      'statistic': server_status.value,
                                      'num_concurrent_socket': self.num_concurrent_socket}

                    sink.send_multipart([client, msg, jsonapi.dumps({**status_runtime,
                                                                     **self.status_args,
                                                                     **self.status_static}), req_id])
                else:
                    self.logger.info('new encode request\treq id: %d\tsize: %d\tclient: %s' %
                                     (int(req_id), int(msg_len), client))
                    # register a new job at sink
                    sink.send_multipart([client, ServerCmd.new_job, msg_len, req_id])

                    # renew the backend socket to prevent large job queueing up
                    # [0] is reserved for high priority job
                    # last used backennd shouldn't be selected either as it may be queued up already
                    rand_backend_socket = random.choice([b for b in backend_socks[1:] if b != rand_backend_socket])

                    # push a new job, note super large job will be pushed to one socket only,
                    # leaving other sockets free
                    job_id = client + b'#' + req_id

                    push_new_job(job_id, msg, int(msg_len))

        self.logger.info('terminated!')

    

class EvalSink(Process):
    def __init__(self, args, front_sink_addr):
        super().__init__()
        self.port = args.port_out
        self.exit_flag = multiprocessing.Event()
        self.logger = set_logger(colored('SINK', 'green'), args.verbose)
        self.front_sink_addr = front_sink_addr
        self.verbose = args.verbose

    def close(self):
        self.logger.info('shutting down...')
        self.exit_flag.set()
        self.terminate()
        self.join()
        self.logger.info('terminated!')

    def run(self):
        self._run()

    @zmqd.socket(zmq.PULL)
    @zmqd.socket(zmq.PAIR)
    @zmqd.socket(zmq.PUB)
    def _run(self, receiver, frontend, sender):
        receiver_addr = auto_bind(receiver)
        frontend.connect(self.front_sink_addr)
        sender.bind('tcp://*:%d' % self.port)

        pending_jobs = defaultdict(lambda: SinkJob())  # type: Dict[str, SinkJob]

        poller = zmq.Poller()
        poller.register(frontend, zmq.POLLIN)
        poller.register(receiver, zmq.POLLIN)

        # send worker receiver address back to frontend
        frontend.send(receiver_addr.encode('ascii'))

        # Windows does not support logger in MP environment, thus get a new logger
        # inside the process for better compability
        logger = set_logger(colored('SINK', 'green'), self.verbose)
        logger.info('ready')

        while not self.exit_flag.is_set():
            socks = dict(poller.poll())
            if socks.get(receiver) == zmq.POLLIN:
                msg = receiver.recv_multipart()
                job_id = msg[0]
                # parsing job_id and partial_id
                job_info = job_id.split(b'@')
                job_id = job_info[0]
                partial_id = int(job_info[1]) if len(job_info) == 2 else 0

                if msg[3] == ServerCmd.data_score:
                    x = jsonapi.loads(msg[1])
                    pending_jobs[job_id].add_score(x, partial_id)
                elif msg[3] == ServerCmd.data_pair:
                    x = jsonapi.loads(msg[1])
                    pending_jobs[job_id].add_pair(x, partial_id)
                else:
                    logger.error('received a wrongly-formatted request (expected 4 frames, got %d)' % len(msg))
                    logger.error('\n'.join('field %d: %s' % (idx, k) for idx, k in enumerate(msg)), exc_info=True)

                logger.info('collect %s %s (E:%d/T:%d/A:%d)' % (msg[3], job_id,
                                                                pending_jobs[job_id].progress_scores,
                                                                pending_jobs[job_id].progress_pairs,
                                                                pending_jobs[job_id].checksum))

                # check if there are finished jobs, then send it back to workers

                finished = [(k, v) for k, v in pending_jobs.items() if v.is_done]
                for job_info, tmp in finished:
                    client_addr, req_id = job_info.split(b'#')
                    x = tmp.result
                    sender.send_multipart([client_addr, x, req_id])
                    logger.info('send back\tsize: %d\tjob id: %s' % (tmp.checksum, job_info))
                    # release the job
                    tmp.clear()
                    pending_jobs.pop(job_info)

            if socks.get(frontend) == zmq.POLLIN:
                client_addr, msg_type, msg_info, req_id = frontend.recv_multipart()
                if msg_type == ServerCmd.new_job:
                    job_info = client_addr + b'#' + req_id
                    # register a new job
                    pending_jobs[job_info].checksum = int(msg_info)
                    logger.info('job register\tsize: %d\tjob id: %s' % (int(msg_info), job_info))
                elif msg_type == ServerCmd.show_config:
                    time.sleep(0.1)  # dirty fix of slow-joiner: sleep so that client receiver can connect.
                    logger.info('send config\tclient %s' % client_addr)
                    sender.send_multipart([client_addr, msg_info, req_id])


class SinkJob:
    def __init__(self):
        self._pending_scores = []
        self.pairs = []
        self.pairs_ids = []
        self.checksum = 0
        self.final_scores = None
        self.progress_pairs = 0
        self.progress_scores = 0

    def clear(self):
        self._pending_scores.clear()
        self.pairs_ids.clear()
        self.pairs.clear()
        del self.final_scores

    def _insert(self, data, pid, data_lst, idx_lst):
        lo = 0
        hi = len(idx_lst)
        while lo < hi:
            mid = (lo + hi) // 2
            if pid < idx_lst[mid]:
                hi = mid
            else:
                lo = mid + 1
        idx_lst.insert(lo, pid)
        data_lst.insert(lo, data)

    def add_score(self, data, pid):
        def fill_data():
            if pid not in self.final_scores:
                self.final_scores[pid] = []
            self.final_scores[pid].append(data)
            self.progress_scores += progress

        progress = 1
        if not self.checksum:
            self._pending_scores.append((data, pid, progress))
        else:
            if self.final_scores is None:
                self.final_scores = {}
            fill_data()
            while self._pending_scores:
                data, pid, progress = self._pending_scores.pop()
                fill_data()

    def add_pair(self, data, pid):
        progress = len(data)
        self._insert(data, pid, self.pairs, self.pairs_ids)
        self.progress_pairs += progress

    @property
    def is_done(self):
        return self.checksum > 0 and self.checksum == self.progress_scores

    @property
    def result(self):
        x = jsonapi.dumps(self.final_scores)
        return x


class EvalWorker(Process):
    def __init__(self, id, args, worker_address_list, sink_address):
        super().__init__()
        self.worker_id = id
        self.logger = set_logger(colored('WORKER-%d' % self.worker_id, 'yellow'), args.verbose)
        self.daemon = True
        self.exit_flag = multiprocessing.Event()
        self.worker_address = worker_address_list
        self.num_concurrent_socket = len(self.worker_address)
        self.sink_address = sink_address
        self.verbose = args.verbose
        
    def close(self):
        self.logger.info('shutting down...')
        self.exit_flag.set()
        self.terminate()
        self.join()
        self.logger.info('terminated!')

    def run(self):
        self._run()

    @zmqd.socket(zmq.PUSH)
    @zmqd.socket(zmq.PUSH)
    @multi_socket(zmq.PULL, num_socket='num_concurrent_socket')
    def _run(self, sink_score, sink_pair, *receivers):
        # Windows does not support logger in MP environment, thus get a new logger
        # inside the process for better compatibility
        logger = set_logger(colored('WORKER-%d' % self.worker_id, 'yellow'), self.verbose)

        for sock, addr in zip(receivers, self.worker_address):
            sock.connect(addr)

        sink_score.connect(self.sink_address)
        sink_pair.connect(self.sink_address)

        for r in self.scorer(receivers, sink_pair):
            send_scores(sink_score, r['client_id'], r['score'], ServerCmd.data_score)
            logger.info('job done\tsize: %s\tclient: %s' % (r['score'], r['client_id']))

    def scorer(self, socks, sink):
        logger = set_logger(colored('WORKER-%d' % self.worker_id, 'yellow'), self.verbose)

        poller = zmq.Poller()
        for sock in socks:
            poller.register(sock, zmq.POLLIN)

        logger.info('ready and listening!')
        
#        summary = ['this is a cat.']
#        references_text = [['this is a cat.'], ['this is a lovely cat.']]
#        logger.info(rouge_n(summary, references_text, 1, alpha=0))
        
#        system = ['This is test summary']
#        references = ['This is ref summary two','this is test summary']        
#                                
#        score = sent_mover_score(references, system * len(references))
#        logger.info(np.mean(score))

        while not self.exit_flag.is_set():
            events = dict(poller.poll())
            for sock_idx, sock in enumerate(socks):
                if sock in events:
                    client_id, raw_msg = sock.recv_multipart()
                    msg = jsonapi.loads(raw_msg)
                    logger.info('new job\tsocket: %d\tsize: %d\tclient: %s' % (sock_idx, len(msg), client_id))
                    # check if msg is a list of list, if yes consider the input is already tokenized

                    for index,_ in enumerate(msg):
                        system = msg[index][0]
                        references = msg[index][1]
                        eval_metric = msg[index][2]
                        if eval_metric[:-1] == 'rouge_':
                            if eval_metric[-1] == '1' or eval_metric[-1] == 'n':
                                score = rouge_n(system, [[r] for r in references], 1, alpha=0)
                            if eval_metric[-1] == '2':
                                score = rouge_n(system, [[r] for r in references], 2, alpha=0)                         
                        if eval_metric[:-1] == 'wmd_':
                            if eval_metric[-1] == '1':
                                score = np.mean(word_mover_score(1, references, system * len(references)))
                            if eval_metric[-1] == '2':
                                score = np.mean(word_mover_score(2, references, system * len(references)))
                        if eval_metric == 'smd':
                            score = np.mean(sent_mover_score(references, system * len(references)))
                            
                        yield {
                            'client_id': client_id,
                            'score': score
                        }

class ServerStatistic:
    def __init__(self):
        self._hist_client = defaultdict(int)
        self._hist_msg_len = defaultdict(int)
        self._client_last_active_time = defaultdict(float)
        self._num_data_req = 0
        self._num_sys_req = 0
        self._num_total_seq = 0
        self._last_req_time = time.perf_counter()
        self._last_two_req_interval = []
        self._num_last_two_req = 200

    def update(self, request):
        client, msg, req_id, msg_len = request
        self._hist_client[client] += 1
        if ServerCmd.is_valid(msg):
            self._num_sys_req += 1
            # do not count for system request, as they are mainly for heartbeats
        else:
            self._hist_msg_len[int(msg_len)] += 1
            self._num_total_seq += int(msg_len)
            self._num_data_req += 1
            tmp = time.perf_counter()
            self._client_last_active_time[client] = tmp
            if len(self._last_two_req_interval) < self._num_last_two_req:
                self._last_two_req_interval.append(tmp - self._last_req_time)
            else:
                self._last_two_req_interval.pop(0)
            self._last_req_time = tmp

    @property
    def value(self):
        def get_min_max_avg(name, stat):
            if len(stat) > 0:
                return {
                    'avg_%s' % name: sum(stat) / len(stat),
                    'min_%s' % name: min(stat),
                    'max_%s' % name: max(stat),
                    'num_min_%s' % name: sum(v == min(stat) for v in stat),
                    'num_max_%s' % name: sum(v == max(stat) for v in stat),
                }
            else:
                return {}

        def get_num_active_client(interval=180):
            # we count a client active when its last request is within 3 min.
            now = time.perf_counter()
            return sum(1 for v in self._client_last_active_time.values() if (now - v) < interval)

        parts = [{
            'num_data_request': self._num_data_req,
            'num_total_seq': self._num_total_seq,
            'num_sys_request': self._num_sys_req,
            'num_total_request': self._num_data_req + self._num_sys_req,
            'num_total_client': len(self._hist_client),
            'num_active_client': get_num_active_client()},
            get_min_max_avg('request_per_client', self._hist_client.values()),
            get_min_max_avg('size_per_request', self._hist_msg_len.keys()),
            get_min_max_avg('last_two_interval', self._last_two_req_interval),
            get_min_max_avg('request_per_second', [1. / v for v in self._last_two_req_interval]),
        ]

        return {k: v for d in parts for k, v in d.items()}
