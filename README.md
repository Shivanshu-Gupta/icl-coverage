# Coverage-based Example Selection

This is the repository for the paper [Coverage-based Example Selection for In-Context Learning](https://arxiv.org/abs/2305.14907). The documentation is WIP but this should help you get started.

The code is organized as follows:

- `data/` contains the datasets used in the paper.
- `src/params.py` defines experiment parameters
- `src/driver.py` is the main file to run a single experiment. Instead of directly running this file, use `src/experiments.py` -- it defines default parameters and makes it easy to run multiple experiments.
- `src/experiments.py` contains the code to run experiments, track experiment statuses and aggregate results. Instead, of directly it dumps the parameters for all the experiments to a file that is then used by `src/run.py`.
- `src/run.py` used to run one or more experiments sequentially or in parallel on one or more gpus. It is the main file to run experiments.

Experiment results are dumped in `results/`.

A typical workflow is as follows:

1. This generates the parameters for 8-shot ICL with all the datasets and LLMs used in the paper and dumps them to `params/params-all.jsonl`.

```bash
python experiments.py --label 'final' \
--datasets "overnight;atis;smcalflow-cs;geoquery;break;mtop" --seeds '0'
--selectors "random;cosine;bm25;bertscore;bertscore_coverage" \
--lms "cushman;codex;starcoder;neo;llama7B;llama13B" \
--lm-batch-size 20 --batch-size 20 --n-shots '8' \
--baselines-exp --paramsfile "params/params-all.jsonl" --run \
--no-collate-results
```

2. This runs the experiments in `params/params-all.jsonl` parallelly on gpus 0 and 1.

```bash
python run.py run-exps-parallel --paramsfile "params/davinci.jsonl" --gpus "0,1"
```
