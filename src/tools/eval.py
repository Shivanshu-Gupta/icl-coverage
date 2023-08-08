from functools import partial
from langchain.prompts.base import BasePromptTemplate

def prompt_to_demos(prompt, example_template: BasePromptTemplate = None):
    prompt = prompt[prompt.find('Sentence: '):]
    demos = prompt.split('\n\n')[:-1]
    def undo_example_template(demo_str):
        source_str, target_str = demo_str.split('\n')
        source = source_str[source_str.find(': ') + len(': '):]
        target = target_str[target_str.find(': ') + len(': '):]
        return dict(source=source, target=target)
    undo_fn = example_template.undo_format if example_template else undo_example_template
    return [undo_fn(d) for d in demos]

def prompt_coverage(res, substruct_fns, example_template, n_shots:int = None):
    from selector.base import bag_relevance
    coverage = {}
    demos = prompt_to_demos(res['prompt'], example_template)
    demo_sources = [d['source'] for d in demos]
    demo_targets = [d['target'] for d in demos]
    test_source = res[example_template.input_variables[0]].strip()
    test_target = res[example_template.input_variables[1]].strip()
    assert n_shots is None or len(demos) == n_shots
    assert test_source == res['prompt'].split('\n\n')[-1].split('\n')[0][len('Sentence: '):]
    for substruct, substruct_fn in substruct_fns.items():
        if 'lf' not in 'substruct':
            test_bag = substruct_fn([test_source])[0]
            demos_bag = set([s for bag in substruct_fn(demo_sources) for s in bag])
        else:
            test_bag = substruct_fn([test_target])[0]
            demos_bag = set([s for bag in substruct_fn(demo_targets) for s in bag])
        coverage[f'{substruct}_recall'] = bag_relevance(test_bag, demos_bag, 'recall')
    return coverage

def get_substruct_fns():
    from tools.structure.substructs import get_parser, get_substructs
    # from selector.base import SelectorUtilsMixin, StructuralSelectorArgs as Args, get_parser, bag_relevance
    # get_substructs = SelectorUtilsMixin.get_substructs
    get_args = lambda substruct, size: dict(substruct=substruct, subst_size=size)
    substruct_fns = {
        'ngram_1': partial(get_substructs, **get_args('ngram', 4)),
        'ngram_4': partial(get_substructs, **get_args('ngram', 4)),
        'depst_4': partial(get_substructs, **get_args('depst', 4), parser=get_parser('spacy')),
        'lfst_4': partial(get_substructs, **get_args('lfst', 4))
    }
    return substruct_fns

def lf_unigram_coverage(res, metric='f1'):
    from langchain.prompts.example_selector.coverage.ast_parser import tokenize_lf
    pred_bag = set(tokenize_lf(res['pred']))
    target_bag = set(tokenize_lf(res['target']))
    common = pred_bag & target_bag
    recall = len(common) / len(target_bag)
    if metric == 'recall': return recall
    precision = len(common) / len(pred_bag)
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def ftk(res):
    from langchain.prompts.example_selector.coverage.ast_parser import target_to_ast
    from tools.structure.ftk import normalized_ftk
    try:
        pred_ast = target_to_ast(res['pred'])
    except:
        return 0
    target_ast = target_to_ast(res['target'])
    return normalized_ftk(target_ast, pred_ast)

def em(res):
    return res['pred'] == res['target']
