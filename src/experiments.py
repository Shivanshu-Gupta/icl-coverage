import json
import jsonlines
import queue
import numpy as np
import pandas as pd
from typing import Optional
from typer import Typer
from copy import deepcopy
from functools import partial
from itertools import product
from pathlib import Path
from rich import print

from params import AllParams, ExperimentParams, LLMParams, sel2cls
from constants import Dataset as D, ExSel as ES, LMType as P, LLM, max_new_tokens_d, test_subset_datasets
from tools.exp import get_strings, get_ints, get_datasets, get_exsels, get_lms
from data_params import ds2cls

app = Typer()
q = queue.Queue()

def process_params(params_l: list[AllParams], only_prompts, only_incomplete, print_only, run, paramsfile):
    print(f'Total {len(params_l)} experiments...')
    params_to_run: list[AllParams] = []
    for i, params in enumerate(params_l):
        if only_incomplete:
            if not only_prompts and params.get_resultsfile().exists():
                print(f'Skipping experiment {i+1}/{len(params_l)}: {params.get_exp_path()} ...')
                continue
            if only_prompts and params.get_promptsfile().exists():
                print(f'Skipping experiment {i+1}/{len(params_l)}: {params.get_exp_path()} ...')
                continue

        params_to_run.append(params)

    print(f'Running {len(params_to_run)} experiments...')
    if print_only:
        for i, params in enumerate(params_to_run):
            if print_only == 'params':
                print(f'\n{i+1}/{len(params_to_run)}:', params)
            if print_only == 'exp_path':
                print(f'\n{i+1}/{len(params_to_run)}:', params.get_exp_path())
            if print_only == 'commands':
                print(f'\n{i+1}/{len(params_to_run)}:', params.get_cmd())
            if print_only == 'logfiles':
                print(f'{i+1}/{len(params_to_run)}:', params.get_logfile())
    elif run:
        with jsonlines.open(paramsfile, mode='w') as writer:
            # breakpoint()
            writer.write_all([p.to_dict() for p in params_to_run])

def compute_coverage_metrics(params_l: list[AllParams], progress=False):
    import shutil, json
    from tools.track import track
    from constants import Dataset as D
    from driver import get_templates
    from eval import get_substruct_fns, prompt_coverage

    all_substruct_fns = get_substruct_fns()
    coverage_metrics_l = []
    for params in track(params_l, disable=not progress):
        resultsfile = params.get_resultsfile()
        if not resultsfile.exists():
            coverage_metrics_l.append(None)
            continue
        example_template = get_templates(
            params.dataset, params.prompt_format, params.input_feature, params.target_feature)['example_template']
        results = json.load(resultsfile.open())
        if params.dataset in [D.GEOQUERY, D.OVERNIGHT]:
            substruct_fns = all_substruct_fns
        else:
            substruct_fns = {k: v for k, v in all_substruct_fns.items() if 'lfst' not in k}
        coverage_metrics = {f'{k}_recall': 0 for k in substruct_fns.keys()}
        if 'coverage' in results and results['coverage'].keys() == coverage_metrics.keys():
            coverage_metrics_l.append(results['coverage'])
            continue
        for res in track(results['results']):
            if 'coverage' not in res:
                res['coverage'] = {}
            if res['coverage'].keys() != coverage_metrics.keys():
                missing_substruct_fns = {k:v for k, v in substruct_fns.items() if f'{k}_recall' not in res['coverage']}
                if missing_substruct_fns:
                    coverage = prompt_coverage(
                        res, missing_substruct_fns, example_template, params.n_shots)
                    res['coverage'] |= coverage
            for k, v in res['coverage'].items():
                coverage_metrics[k] += v / len(results['results'])
        results['coverage'] = coverage_metrics
        coverage_metrics_l.append(coverage_metrics)
        shutil.move(resultsfile, f'{resultsfile}.bak.2')
        json.dump(results, resultsfile.open('w'), indent=2)
    return coverage_metrics_l

def get_single_results(i, N, P: AllParams, train_results=False, coverage_results=False, train_states=False):
    # finalresults = P.to_flattened_dict()
    finalresults = {k.split('.')[-1]: v for k, v in P.to_flattened_dict().items()}
    if P.selector.selector_type not in [ES.RANDOM]:
        finalresults |= dict(selector_name=P.get_selector_name())
    if train_results:
        historyfile = P.get_ckpt_dir() / 'history.json'
        if historyfile.exists():
            finalresults |= json.load(historyfile.open())
            if 'epoch_accuracies' in finalresults:
                finalresults |= dict(
                    train_accuracy=100*max(finalresults['epoch_accuracies']),
                    train_results=True)
            else:
                finalresults |= dict(
                    train_accuracy=100*max(finalresults['train_accuracy_l']),
                    val_accuracy=max(finalresults['val_accuracy_l']),
                    train_results=True)
        statesfile = P.get_ckpt_dir() / 'states.json'
        if train_states and statesfile.exists():
            states = json.load(statesfile.open())
            finalresults["states"] = states
    logfile = P.get_logfile()
    resultsfile = P.get_resultsfile()
    promptsfile = P.get_promptsfile()
    lastlog = ''
    if logfile.exists():
        try:
            lines = open(logfile).readlines()
            if lines: lastlog = lines[-1].strip()
        except:
            pass
    print(f'{i+1}/{N}', resultsfile, resultsfile.exists(), promptsfile.exists(), lastlog)
    if resultsfile.exists():
        results = json.load(resultsfile.open())
        metrics = results['metrics'] if 'metrics' in results else results
        accuracy = metrics['accuracy' if 'accuracy' in metrics else 'overall_accuracy']
        finalresults |= dict(test_accuracy=accuracy, completed=True)
        for m in ['lfem', 'bleu', 'ngram_1_recall', 'ngram_4_recall', 'depst_4_recall', 'lfst_4_recall']:
            if m in metrics: finalresults |= {m: metrics[m]}
        if 'n_shots' in metrics:
            finalresults |= dict(avg_n_shots=metrics['n_shots'])
        if coverage_results:
            if not 'coverage' in metrics:
                print('Computing coverage metrics...')
                metrics['coverage'] = compute_coverage_metrics([P])[0]
            finalresults |= metrics['coverage']
    return finalresults

def get_results(params_l: list[AllParams], train_results=False, coverage_results=False, train_states=False, parallel=False):
    from joblib import Parallel, delayed
    if parallel:
        with Parallel(n_jobs=20, verbose=True) as parallel:
            results_l = parallel(delayed(get_single_results)(
                    i, len(params_l), params, train_results, coverage_results, train_states)
                for i, params in enumerate(params_l))
    else:
        results_l = [get_single_results(i, len(params_l), params, train_results, coverage_results, train_states)
                    for i, params in enumerate(params_l)]
    if not results_l: return None
    resultsdf = pd.DataFrame.from_records(results_l)
    return resultsdf

if True:    # Params
    lm_args_d = {
        'openai': dict(
            lm_type=P.OPENAI, lm_url=None,
            lm_name=[LLM.CODE_CUSHMAN_001, LLM.CODE_DAVINCI_002],
            lm_batch_size=7, lm_delay=10,),
        'cushman': dict(
            lm_name=LLM.CODE_CUSHMAN_001, lm_type=P.OPENAI, lm_url=None,
            lm_batch_size=7, lm_delay=2, openai_keys_file='../../codex_keys.txt'),
        'codex': dict(
            lm_name=LLM.CODE_DAVINCI_002, lm_type=P.OPENAI, lm_url=None,
            lm_batch_size=7, lm_delay=2, openai_keys_file='../../codex_keys.txt'),
        'turbo': dict(
            lm_name=LLM.TURBO, lm_type=P.OPENAI_CHAT, lm_url=None,
            lm_batch_size=1, lm_delay=10,),
        'davinci': dict(
            lm_name=LLM.TEXT_DAVINCI_003, lm_type=P.OPENAI, lm_url=None,
            lm_batch_size=7, lm_delay=1,),
        'opt': dict(
            lm_name=LLM.OPT_30B, lm_type=P.OPT_SERVER, lm_batch_size=7, lm_delay=10,
            lm_url='http://ava-s1.ics.uci.edu:8890',),
        'neo': dict(
            lm_name=LLM.NEO, lm_type=P.HUGGINGFACE, do_sample=False, lm_batch_size=10,),
        'jt6b': dict(
            lm_name=LLM.JT6B, lm_type=P.HUGGINGFACE, do_sample=False, lm_batch_size=7,),
        'neox': dict(
            lm_name=LLM.NEOX20B, lm_type=P.HUGGINGFACE, do_sample=False, lm_batch_size=2,),
        'llama7B': dict(
            lm_name=LLM.LLAMA7B, lm_type=P.HUGGINGFACE, do_sample=False, lm_batch_size=7,),
        'llama13B': dict(
            lm_name=LLM.LLAMA13B, lm_type=P.HUGGINGFACE, do_sample=False, lm_batch_size=7,),
        'starcoder': dict(
            lm_name=LLM.STARCODER, lm_type=P.HUGGINGFACE, do_sample=False, lm_batch_size=7,),
    }

    def common_dataset_args(dataset, **overrides):
        if dataset in max_new_tokens_d:
            return dict(dataset=dataset, max_tokens=max_new_tokens_d[dataset], prefix=False) | overrides
        else:
            return dict(dataset=dataset, prefix=False) | overrides

    dataset_args_d: dict[D, dict] = {
        D.SMCALFLOW_CS: dict(
            dataset=D.SMCALFLOW_CS,
            input_feature=['source', 'paraphrase'][:1],
            target_feature=['target', 'original_target'][:1],
            max_tokens=256, prefix=False),
        D.GSM8K: common_dataset_args(D.GSM8K, prefix=True),
        D.COGS: common_dataset_args(D.COGS, test_split=['gen', 'dev']),
    }
    def get_dataset_args(dataset):
        if dataset in dataset_args_d:
            return dataset_args_d[dataset]
        else:
            return common_dataset_args(dataset=dataset)

    selector_args_d: dict[str, tuple[ES, dict]] = {
        'random': dict(selector_type=ES.RANDOM),
        'cosine': dict(selector_type=ES.COSINE, coverage=False),
        'cosine_coverage': dict(selector_type=ES.COSINE, coverage=True, reorder=[False, True]),
        'recall': dict(selector_type=ES.STRUCT, metric='recall', coverage=False),
        'recall_coverage': dict(selector_type=ES.STRUCT,
            metric='recall', coverage=True, ordering=[None, 'recall'][1]),
        'bm25': dict(selector_type=ES.STRUCT, metric='bm25', coverage=False),
        'bm25_coverage': dict(selector_type=ES.STRUCT,
            metric='bm25', coverage=True, ordering=[None, 'bm25'], add_cand_score=False),
        'bm25_coverage_candscore': dict(selector_type=ES.STRUCT,
            metric='bm25', coverage=True, ordering=[None, 'bm25'], add_cand_score=True,
            cand_score_discount=[1, 3]),
        'bertscore': dict(selector_type=ES.BERTSCORE, metric='recall', coverage=False),
        'bertscore_prec': dict(selector_type=ES.BERTSCORE, metric=['precision', 'f1'], coverage=False),
        'set_bsr': dict(selector_type=ES.BERTSCORE,
            metric='recall', coverage=True, add_cand_score=[False, True][:1]),
        'lf_coverage': dict(selector_type=ES.LF_COVERAGE),
    }

@app.command()
def main(
    label: str = 'exp0',
    data_root: Path = Path('../data'),
    output_root: Path = Path('../results'),
    seeds: str = '0;1;2;3;4', only_incomplete: bool = False,
    baselines_exp: Optional[bool] = False,
    gpu: int = 0, debug: bool = False, tiny: bool = False, only_prompts: bool = False,
    datasets: str = 'geoquery', lms: str = 'codex', selectors: str = 'random',
    return_params: bool = False, print_only: str = '',
    run: bool = False, paramsfile: Path = Path('params.jsonl'),
    collate_results: bool = True, train_results: bool = False, coverage_results: bool = False,
    batch_size: int = 28, lm_batch_size: Optional[int] = None,
    n_shots: Optional[str] = None, n_cands: Optional[str] = None, prompt_version: Optional[str] = None,
):
    overrides = dict(exp={}, data={}, llm={}, selector={})
    if lm_batch_size: overrides['llm']['lm_batch_size'] = lm_batch_size
    if n_shots: overrides['selector']['n_shots'] = get_ints(n_shots)
    if n_cands: overrides['data']['n_cands'] = get_ints(n_cands)
    if prompt_version: overrides['data']['prompt_version'] = prompt_version

    geoquery_splits = [
        'iid', 'csl_length',
        *[f'csl_template_{i}' for i in range(1, 4)],
        *[f'csl_tmcd_{i}' for i in range(1, 4)],
    ]

    ds2splits = {
        D.OVERNIGHT: ['socialnetwork_iid_0', 'socialnetwork_template_0'],
        D.ATIS: ['iid_0', 'template_0'],
        D.GEOQUERY: geoquery_splits,
        # D.SMCALFLOW_CS: ['0_S', '8_S', '0_C', '8_C', '16_C', '32_C'],
        D.SMCALFLOW_CS: ['8_S', '32_C'],
    }

    def get_params_l(
        seed, dataset, lm, selector, n_cands=-1, selector_args={}, splits=None
    ):
        exp_args = dict(
            label=label, data_root=data_root, output_root=output_root,
            gpu=gpu, debug=debug, tiny=tiny, only_prompts=only_prompts,
            batch_size=batch_size, seed=seed)
        dataset_args = get_dataset_args(dataset) | dict(
            n_cands=n_cands,
            n_test=1000 if dataset in test_subset_datasets else -1,
        ) | overrides['data']
        lm_args = deepcopy(lm_args_d[lm]) | overrides['llm']
        selector_args = selector_args_d[selector] | selector_args | overrides['selector']
        selector_type = selector_args['selector_type']

        def _get_params_l(exp_args, dataset_args, lm_args, selector_args):
            return AllParams(
                exp=ExperimentParams(**exp_args),
                data=ds2cls[dataset](**dataset_args),
                llm=LLMParams(**lm_args),
                selector=sel2cls[selector_type](**selector_args)
            ).get_settings()

        if splits:
            params_l: list[AllParams] = []
            for split in splits:
                dataset_args |= dict(split=split)
                if dataset == D.GEOQUERY and seed > 0 and 'lf' not in selector \
                    and split is not None and ('csl_template' in split or 'csl_tmcd' in split):
                    continue
                params_l += _get_params_l(exp_args, dataset_args, lm_args, selector_args)
            return params_l
        else:
            return _get_params_l(exp_args, dataset_args, lm_args, selector_args)

    cosine_emb_lms = ['bert-base-uncased', 'sentence-transformers/all-mpnet-base-v2'][1:]
    bertscore_emb_lms = ['microsoft/deberta-base-mnli', 'microsoft/deberta-large-mnli']
    params_l: list[AllParams] = []
    for seed, dataset, lm, selector in product(
        get_ints(seeds), get_datasets(datasets), get_strings(lms), get_strings(selectors)
    ):
        splits = ds2splits.get(dataset, None)
        common = [seed, dataset, lm, selector]
        get_params_fn = partial(get_params_l, *common, splits=splits)
        if selector == 'random':
            params_l += get_params_fn()
        elif selector in ['cosine', 'cosine_coverage']:
            emb_lms = 'sentence-transformers/all-mpnet-base-v2' if baselines_exp else cosine_emb_lms
            params_l += get_params_fn(selector_args=dict(emb_lm=emb_lms))
        elif selector == 'bertscore':
            idfs = False if baselines_exp else [True, False]
            emb_lms = 'microsoft/deberta-large-mnli' if baselines_exp else bertscore_emb_lms
            params_l += get_params_fn(selector_args=dict(idf=idfs, emb_lm=emb_lms))
        elif selector == 'bertscore_prec':
            emb_lms = 'microsoft/deberta-large-mnli' if baselines_exp else bertscore_emb_lms
            params_l += get_params_fn(selector_args=dict(idf=False, emb_lm=emb_lms))
        elif selector == 'set_bsr':
            idfs = False if baselines_exp else [True, False]
            emb_lms = 'microsoft/deberta-large-mnli' if baselines_exp else bertscore_emb_lms
            orderings = 'recall' if baselines_exp else [None, 'recall']
            params_l += get_params_fn(selector_args=dict(idf=idfs, emb_lm=emb_lms, ordering=orderings))
        elif selector in ['recall', 'bm25', 'recall_coverage', 'bm25_coverage', 'bm25_coverage_candscore']:
            if baselines_exp:
                if selector == 'bm25': params_l += get_params_fn(
                    selector_args=dict(substruct='ngram', subst_size=1))
                elif selector == 'bm25_coverage': params_l += get_params_fn(
                    selector_args=dict(substruct='ngram', ordering='bm25', subst_size=4))
                else: continue
            else:
                if dataset != D.NL2BASH: params_l += get_params_fn(
                    selector_args=dict(substruct='depst', subst_size=4))
                params_l += get_params_fn(selector_args=dict(substruct='ngram', subst_size=[1, 4]))
        elif selector == 'lf_coverage':
            if dataset == D.SMCALFLOW_CS: splits = ['8_S', '32_C']
            elif dataset == D.GEOQUERY: splits = geoquery_splits
            else: continue
            params_l += [
                *get_params_l(0, *common[1:], splits=splits),
                *get_params_l(1, *common[1:], splits=splits),
                *get_params_l(2, *common[1:], splits=splits),
            ]

    from collections import Counter
    freqs = Counter([p.get_exp_path() for p in params_l])
    if freqs.most_common(1)[0][1] > 1:
        print('WARNING: duplicate params')
        print(freqs.most_common(5))

    if return_params: return params_l
    process_params(params_l, only_prompts, only_incomplete, print_only, run, paramsfile)
    if collate_results:
        resultsdf: pd.DataFrame = get_results(
            params_l, train_results=train_results, coverage_results=coverage_results, train_states=False)
        if resultsdf is not None and resultsdf.completed.any():
            make_tables(resultsdf, output_root / label / f'{lm}-1', aggregate_csl=False)
            # make_tables(resultsdf, output_root / label / f'{lm}-2', aggregate_csl=True)
        return resultsdf


def make_tables(resultsdf: pd.DataFrame, output=True, resultsfile=None, aggregate_csl=False, count=True, fillna=True):
    resultsdf = resultsdf[resultsdf.completed]
    filter_cols = lambda cols, allowed: [c for c in cols if c in allowed]
    common_cols = ['dataset', 'input_feature', 'split', 'test_split', 'n_test',
                   'prompt_format', 'n_shots', 'lm_name', 'selector_type', 'selector_name']
    similar_cols = ['emb_lm', 'sim_metric']
    rl_cols = [
        'n_cands', 'embedding_size', 'n_linear', 'one_each',
        'init_ortho', 'n_layers', 'policy_temp', 'log_1mprob', 'sample_train',
        'diversity_loss', 'diversity_loss_agg', 'diversity_loss_weight']
    struct_cols = ['substruct', 'subst_size', 'ordering', 'selector_metric', 'coverage',]
    coverage_cols = ['n_combs', 'greedy_coverage', 'depparser',
                     'template_diversity', 'use_paraphrase', 'break_on_reset']
    bertscore_cols = ['idf', 'embed_context']
    cosine_cols = ['reorder']
    columns = filter_cols(
        [*common_cols, *similar_cols, *rl_cols, *struct_cols, *coverage_cols, *bertscore_cols, *cosine_cols],
        resultsdf.columns)
    resultsdf = resultsdf.sort_values(columns)

    if aggregate_csl:
        def csl(row):
            if row.split is not None and (row.split.startswith('csl_tmcd') or row.split.startswith('csl_template')):
                parts = row.split.split('_')
                return pd.Series(['_'.join(parts[:2]), parts[2]])
            else:
                return pd.Series([row.split, None])
        resultsdf[['split', 'csl_seed']] = resultsdf.apply(csl, axis=1)

    resultsdf[columns] = resultsdf[columns].replace(-1, 'all', regex=True)

    for col in columns:
        if col not in ['dataset', 'split'] and len([v for v in resultsdf[col].unique() if v is not None]) <= 1:
            resultsdf = resultsdf.drop(col, axis=1)

    remaining_columns = filter_cols(columns, resultsdf.columns)
    metric_cols = filter_cols(
        ['test_accuracy', 'train_accuracy', 'val_accuracy',
         'ngram_1_recall', 'ngram_4_recall', 'depst_4_recall', 'lfst_4_recall',
         'bleu', 'lfem', 'avg_n_shots'],
        resultsdf.columns)
    final_df: pd.DataFrame = resultsdf.groupby(remaining_columns, dropna=False).agg({
        'test_accuracy': ['mean', 'count'][:1] if count else 'mean',
        **{c: 'mean' for c in metric_cols if c != 'test_accuracy'}})
    if fillna:
        def fillna_index(df, my_fillna_value):
            if isinstance(df.index, pd.MultiIndex):
                df.index = pd.MultiIndex.from_frame(
                    df.index.to_frame().fillna(my_fillna_value)
                )
            else:
                df.index = df.index.fillna(my_fillna_value)
            return df

        final_df = fillna_index(final_df, '-')
    if output:
        with pd.option_context(
            'display.max_rows', None,
            # 'display.max_columns', None,
            "display.max_colwidth", 70,
            'display.precision', 2,
        ):
            print(final_df)
            summary_df = final_df.reset_index().droplevel(level=1, axis=1)
            params_cols = filter_cols(
                ['dataset', 'split', 'test_split', 'lm_name', 'selector_type', 'selector_name', 'coverage'],
                summary_df.columns)
            metric_cols = filter_cols(['test_accuracy', 'lfem'], summary_df.columns)
            if params_cols and metric_cols:
                summary_df = summary_df[[*params_cols, *metric_cols]].set_index(params_cols)
                print(summary_df)
    if resultsfile:
        final_df.to_latex(f'{resultsfile}.tex', escape=False, multirow=True, multicolumn=True, float_format='%.2f', column_format='l' + 'r' * (len(final_df.columns) - 1))
        final_df.to_excel(f'{resultsfile}.xlsx', index=True, merge_cells=True, float_format='%.2f')
    return final_df

if __name__ == '__main__':
    app()