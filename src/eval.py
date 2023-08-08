import json
import math
import numpy as np
from contextlib import nullcontext
from functools import partial
from rich.live import Live

from langchain.chains import LLMChain
from langchain.llms import BaseLLM

from params import AllParams
from constants import Dataset as D
from tools.utils import Logger
from tools.track import get_progress_bar
from constants import lfst_prompt_cov_datasets, no_prompt_cov_datasets, generation_datasets, mwp_datasets, LLM, Dataset as D
from prompts.base import ExampleTemplate
from prompts.few_shot import FewShotPromptTemplate2

def prompt_to_demos(prompt, prefix, example_template: ExampleTemplate = None):
    if prefix.format():
        demos = prompt.split('\n\n')[1:-1]
    else:
        demos = prompt.split('\n\n')[:-1]
    def undo_example_template(demo_str):
        source_str, target_str = demo_str.split('\n')
        source = source_str[source_str.find(': ') + len(': '):]
        target = target_str[target_str.find(': ') + len(': '):]
        return dict(source=source, target=target)
    undo_fn = example_template.undo_format if example_template else undo_example_template
    return [undo_fn(d) for d in demos]

def prompt_coverage(ex, substruct_fns, prefix, example_template, prompt=None, demos=None, n_shots:int = None):
    from selector.base import bag_relevance
    coverage = {}
    if demos is None:
        demos = prompt_to_demos(prompt, prefix, example_template)
        demo_sources = [d['source'] for d in demos]
        demo_targets = [d['target'] for d in demos]
    else:
        demo_sources = [example_template.get_source(**d) for d in demos]
        demo_targets = [example_template.get_target(**d) for d in demos]
    test_source = example_template.get_source(**ex)
    test_target = example_template.get_target(**ex)
    assert n_shots is None or len(demos) == n_shots
    # assert test_source == prompt.split('\n\n')[-1].split('\n')[0][len('Sentence: '):], prompt
    for substruct, substruct_fn in substruct_fns.items():
        if 'lf' not in 'substruct':
            test_bag = substruct_fn([test_source])[0]
            demos_bag = set([s for bag in substruct_fn(demo_sources) for s in bag])
        else:
            test_bag = substruct_fn([test_target])[0]
            demos_bag = set([s for bag in substruct_fn(demo_targets) for s in bag])
        coverage[f'{substruct}_recall'] = 100 * bag_relevance(test_bag, demos_bag, 'recall')
    return coverage

def get_substruct_fns(lfst:bool = True):
    from tools.structure.substructs import get_parser, get_substructs
    # from selector.base import SelectorUtilsMixin, StructuralSelectorArgs as Args, get_parser, bag_relevance
    # get_substructs = SelectorUtilsMixin.get_substructs
    get_args = lambda substruct, size: dict(substruct=substruct, subst_size=size)
    substruct_fns = {
        'ngram_1': partial(get_substructs, **get_args('ngram', 1)),
        'ngram_4': partial(get_substructs, **get_args('ngram', 4)),
        'depst_4': partial(get_substructs, **get_args('depst', 4), parser=get_parser('spacy')),
    }
    if lfst:
        substruct_fns['lfst_4'] = partial(get_substructs, **get_args('lfst', 4))
    return substruct_fns

def aggregate_metrics(agg_metrics, ex_metrics):
    for k, v in ex_metrics.items():
        agg_metrics[k] = agg_metrics.get(k, 0) + v

def eval(
    params: AllParams, test_ds, llm: BaseLLM,
    prompt_template: FewShotPromptTemplate2,
    batch_size, logger: Logger,
    outfile=None, progress=None, debug=False
):
    log = logger.log
    n_test, n_batch = len(test_ds), math.ceil(len(test_ds) / batch_size)
    tokenizer = llm.tokenizer if hasattr(llm, 'tokenizer') else None
    substruct_fns = get_substruct_fns(lfst=params.dataset in lfst_prompt_cov_datasets)
    example_template = prompt_template.example_template
    sep = prompt_template.example_separator
    prompt_coverage_fn = partial(
        prompt_coverage,
        substruct_fns=substruct_fns,
        prefix=prompt_template.prefix_template,
        example_template=example_template
    )
    results, agg_metrics = [], {}
    n_correct, n_total = 0, 0
    # debug = True
    progress = progress or get_progress_bar(console=logger.std_console)
    with Live(progress, refresh_per_second=1, console=logger.std_console) if not debug else nullcontext():
        for batch_i in progress.track(range(n_batch), description='Evaluating..') if not debug else range(n_batch):
            log(f"Batch {batch_i+1}/{n_batch}")
            test_batch = test_ds.select(np.arange(
                batch_i * batch_size, min(len(test_ds), (batch_i + 1) * batch_size)))
            prompts, demos_l = zip(*[prompt_template.format(**ex, return_demos=True)
                                      for ex in test_batch])
            if params.dataset in generation_datasets:
                if params.lm_name != LLM.TURBO or params.dataset not in [D.GSM8K, D.DROP]:
                    response = llm.generate(prompts, stop=[sep])
                    llm_outputs = [gen[0].text for gen in response.generations]
                else:
                    messages = [[example_template.prepare_for_turbo(e)
                                 for e in p.split(sep)] for p in prompts]
                    response = llm._generate(messages, stop=[sep])
                    llm_outputs = [gen[0].text for gen in response.generations]
            else:   # classification
                if params.lm_name == LLM.TURBO:
                    raise NotImplementedError
                llm_outputs = llm.classify_v2(prompts=prompts, choices=example_template.get_choices())

            for ex, prompt, demos, llm_output in zip(test_batch, prompts, demos_l, llm_outputs):
                prompt_metrics = dict(n_shots=len(demos))
                if params.dataset not in no_prompt_cov_datasets:
                    prompt_metrics |= prompt_coverage_fn(ex=ex, demos=demos)
                orig_prompt = prompt
                res = dict(
                    **ex,
                    # demos=demos,
                    metrics=prompt_metrics
                )
                if tokenizer:
                    res['orig_prompt'] = orig_prompt
                    prompt = tokenizer.decode(tokenizer.encode(prompt), skip_special_tokens=True)
                res['prompt'] = prompt
                res['completion'] = llm_output

                if params.dataset in generation_datasets:
                    res['pred'] = example_template.parse_output(res['completion'].strip(), **ex)
                    target = example_template.get_target(**ex)
                    if tokenizer and not params.dataset in mwp_datasets:
                        target = tokenizer.decode(tokenizer.encode(target), skip_special_tokens=True)
                    res['actual_target'] = target
                    metrics = example_template.check_output(res['pred'], target, **ex)
                else:
                    res['pred'] = res['completion']
                    metrics = example_template.check_output(res['pred'], **ex)
                res['metrics'] |= metrics
                results.append(res)
                aggregate_metrics(agg_metrics, res['metrics'])
                n_correct += res['metrics']['accuracy'] / 100
                n_total += 1
                if debug:
                    log('Prompt and Completion:')
                    log(f"[green]{prompt}[/green][red]{res['completion']}[/red]")
                    log(f'Inputs: {ex}')
            log({k: v / n_total for k, v in agg_metrics.items()})
            acc_str = f'Accuracy: {100 * n_correct / n_total:.2f} ({n_correct} / {n_total})'
            log(acc_str)
            if not debug:
                progress.tasks[-1].description = acc_str

    for k in agg_metrics:
        agg_metrics[k] /= n_total
    agg_metrics |= dict(n_correct=n_correct, n_total=n_total)

    log(ex)
    log(prompt)
    log(f'LLM Output: {llm_output}')
    log(agg_metrics)
    log(f'Final {acc_str}')

    data = dict(results=results, metrics=agg_metrics, **agg_metrics)
    if outfile:
        print(f"Saving results to {outfile} ..")
        json.dump(data, open(outfile, 'w'), indent=2, separators=(',', ': '))
    return data

def dump_prompts(
    params: AllParams, test_ds, prompt_template: FewShotPromptTemplate2, logger: Logger,
    outfile=None, progress=None, debug=False
):
    log = logger.log
    results = []
    progress = progress or get_progress_bar(console=logger.std_console)
    substruct_fns = get_substruct_fns(lfst=params.dataset in lfst_prompt_cov_datasets)
    total_coverage = {f'{k}_recall': 0 for k in substruct_fns.keys()}
    with Live(progress, refresh_per_second=1, console=logger.std_console) if not debug else nullcontext():
        for ex in progress.track(test_ds, description='Creating Prompts..') if not debug else test_ds:
            prompt = prompt_template.format(**ex)
            res = dict(**ex, prompt=prompt)
            coverage = prompt_coverage(res, substruct_fns, prompt_template.prefix_template, prompt_template.example_template)
            res['coverage'] = coverage
            results.append(res)
            total_coverage = {k: v + coverage[k] / len(test_ds) for k, v in total_coverage.items()}
    data = dict(results=results, coverage=total_coverage,)
    if outfile:
        print(f"Saving results to {outfile} ..")
        with open(outfile, 'w') as f:
            json.dump(data, f, indent=2, separators=(',', ': '))

def lf_unigram_coverage(res, metric='f1'):
    from tools.structure.ast_parser import tokenize_lf
    pred_bag = set(tokenize_lf(res['pred']))
    target_bag = set(tokenize_lf(res['target']))
    common = pred_bag & target_bag
    recall = len(common) / len(target_bag)
    if metric == 'recall': return recall
    precision = len(common) / len(pred_bag)
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def ftk(res):
    from tools.structure.ast_parser import target_to_ast
    from tools.structure.ftk import normalized_ftk
    try:
        pred_ast = target_to_ast(res['pred'])
    except:
        return 0
    target_ast = target_to_ast(res['target'])
    return normalized_ftk(target_ast, pred_ast)

def em(res):
    return res['pred'] == res['target']
