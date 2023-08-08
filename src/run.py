import jsonlines
import os
import queue
import subprocess
import shlex
import torch
import gc
from typer import Typer, Option
from pathlib import Path
from joblib import Parallel, delayed, parallel_backend
from rich import print
from tools.exp import get_ints
from params import AllParams
from driver import main

app = Typer()
q = queue.Queue()

def run_exp(P: AllParams):
    import shlex
    import subprocess
    cmd = P.get_cmd()
    print(cmd)
    args = shlex.split(cmd)
    process = subprocess.Popen(args)
    logfile = P.get_logfile()
    os.makedirs(logfile.parent, exist_ok=True)
    print(f'Logging to: {logfile}')
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    tee = subprocess.Popen(['tee', logfile], stdin=process.stdout)
    process.stdout.close()
    tee.communicate()
    ret = process.wait()
    return ret

@app.command()
def run_exps(
    paramsfile: Path = Option('params.jsonl', help='Path to the params file.'),
    # from_start: bool = Option(False, help='Start from the beginning.')
):
    while True:
        if not paramsfile.exists():
            print('Params file does not exist...')
            return

        params = None
        with jsonlines.open(paramsfile, mode='r') as reader:
            params_l = [AllParams.from_dict(p) for p in reader]
            # if from_start:
            #     for p in params_l:
            #         p['completed'] = False
            for idx, p in enumerate(params_l):
                if not p['completed']:
                    params = p
                    break
        if params is None:
            print('No more params to run...')
            return

        print(f'\nRunning experiment {idx+1}/{len(params_l)}: {params.get_exp_path()} ...')
        print(params)
        params = params.to_dict()
        del params['completed']
        try:
            main(**params)
            params_l[idx]['completed'] = True
            # run_exp(params)
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            print(f'Error: {e}')
            print(type(e))
            from openai.error import APIError, APIConnectionError
            if isinstance(e, (APIError, APIConnectionError)):
                print('OpenAI API error, rerunning experiment...')
                params_l[idx]['completed'] = False

        with jsonlines.open(paramsfile, mode='w') as writer:
            writer.write_all([p.to_dict() for p in params_l])

def run_inference(idx, params: AllParams, debug=False):
    gpu = q.get(block=True)
    params.exp.gpu = gpu
    cmd = params.get_cmd()
    outfile = params.get_outfile()
    # print(idx, cmd, '>', logfile)
    print(idx+1, cmd)
    os.makedirs(outfile.parent, exist_ok=True)
    if not debug:
        args = shlex.split(cmd)
        process = subprocess.Popen(args, stdout=outfile.open('w'), stderr=subprocess.STDOUT)
        ret = process.wait()
        torch.cuda.empty_cache()
    q.put(gpu)

@app.command()
def run_exps_parallel(
    paramsfile: Path = Option('params.jsonl', help='Path to the params file.'),
    gpus: str = Option('0,1,2,3,4,5,6,7', help='Comma separated list of GPUs to use.'),
    start_idx: int = Option(0, help='Start from this index.'),
    debug: bool = Option(False, help='Run in debug mode.'),
    clear_logs: bool = Option(False, help='Clear logs.'),
):
    if not paramsfile.exists():
        print('Params file does not exist...')
        return

    with jsonlines.open(paramsfile, mode='r') as reader:
        params_l = [AllParams.from_dict(p) for p in reader][start_idx:]
        # params_l = [p for p in params_l if not p['completed']]
        if clear_logs:
            for p in params_l:
                outfile = p.get_outfile()
                if outfile.exists(): os.remove(outfile)
                logfile = p.get_logfile()
                if logfile.exists(): os.remove(logfile)

    gpus = get_ints(gpus, sep=',')
    for gpu in gpus:
        q.put(gpu)

    n_jobs = len(params_l)
    n_concurrent = len(gpus)
    def train_wrapper(i, params):
        print(f'  > {i+1}/{n_jobs} {params.get_outfile()}')
        run_inference(i, params, debug)
        print(f'  < {i+1}/{n_jobs} {params.get_outfile()}')
    print(f'Running {len(params_l)} jobs...')
    while params_l:
        with Parallel(n_jobs=n_concurrent, require='sharedmem', verbose=True) as parallel:
            parallel(delayed(train_wrapper)(i, params)
                     for i, params in enumerate(params_l))
        if debug: break
        completed_l = [p.get_resultsfile().exists() for p in params_l]
        if not any(completed_l):
            print('all jobs failed')
            break
        params_l = [p for p, c in zip(params_l, completed_l) if not c]
        if params_l:
            print(f'Rerunning {len(params_l)} failed jobs...')
        else:
            print('All jobs completed')

if __name__ == '__main__':
    app()