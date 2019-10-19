def main():
    from summ_eval.server import EvalServer
    from summ_eval.server.helper import get_run_args
    args = get_run_args()
    server = EvalServer(args)
    server.start()
    server.join()