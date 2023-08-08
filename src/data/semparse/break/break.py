# modified from https://huggingface.co/datasets/drop/raw/main/drop.py

import json
import os
import nltk
import ast
import datasets

def make_tree(p):
    # p = ds.program[9]
    # steps = ast.literal_eval(p)
    try:
        ops = [(app[:app.find('[')], ast.literal_eval(app[app.find('['):])) for app in ast.literal_eval(p)]
        trees = [nltk.Tree(op, args) for op, args in ops]
        for (fn, args), t in zip(ops, trees):
            for i, arg in enumerate(args):
                if arg.startswith('#'):
                    idx = int(arg[1:])
                    t[i] = trees[idx-1]
        return trees[-1]
    except:
        return None

def tree_to_tuple(t):
    if isinstance(t, nltk.Tree):
        return (t.label(), tuple([tree_to_tuple(c) for c in t]))
    else:
        return t

def tuple_to_tree(t):
    if isinstance(t, tuple):
        return nltk.Tree(t[0], [tuple_to_tree(c) for c in t[1]])
    else:
        return t

def anonymize_tree(t, anon=None):
    if isinstance(t, nltk.Tree):
        # return nltk.Tree(t.label(), [anonymize_tree(c) for c in t])
        match t.label():
            case 'SELECT':
                assert len(t) == 1
                c0 = anonymize_tree(t[0], anon='_ent_')
                return nltk.Tree(t.label(), [c0])
            case 'FILTER':
                assert len(t) == 2
                c0 = anonymize_tree(t[0], anon='_cond_')
                c1 = anonymize_tree(t[1], anon='_ref_')
                return nltk.Tree(t.label(), [c0, c1])
            case 'PROJECT':
                assert len(t) == 2
                c0 = anonymize_tree(t[0], anon='_rel_')
                c1 = anonymize_tree(t[1], anon='_ref_')
                return nltk.Tree(t.label(), [c0, c1])
            case 'AGGREGATE':
                assert len(t) == 2
                c0 = t[0]
                c1 = anonymize_tree(t[1], anon='_ref_')
                return nltk.Tree(t.label(), [c0, c1])
            case 'GROUP':
                assert len(t) == 3
                c0 = anonymize_tree(t[0])
                c1 = anonymize_tree(t[1], anon='_ref_')
                assert not isinstance(t[2], str)
                c2 = anonymize_tree(t[2], anon='_ref_')
                return nltk.Tree(t.label(), [c0, c1, c2])
            case 'INTERSECTION':
                assert len(t) == 3
                c0 = anonymize_tree(t[0], anon='_rel_')
                c1 = anonymize_tree(t[1], anon='_ref_')
                c2 = anonymize_tree(t[2], anon='_ref_')
                return nltk.Tree(t.label(), [c0, c1, c2])
            case 'UNION':
                return nltk.Tree(t.label(), [anonymize_tree(c) for c in t])
            case 'COMPARATIVE':
                assert len(t) == 3
                c0 = anonymize_tree(t[0], anon='_cond_')
                c1 = anonymize_tree(t[1], anon='_ref_')
                c2 = anonymize_tree(t[2], anon='_ref_')
                return nltk.Tree(t.label(), [c0, c1, c2])
            case 'SUPERLATIVE':
                assert len(t) == 3
                c0 = t[0]
                c1 = anonymize_tree(t[1], anon='_ref_')
                c2 = anonymize_tree(t[2], anon='_ref_')
                return nltk.Tree(t.label(), [c0, c1, c2])
            case 'BOOLEAN':
                if len(t[0].split(' ')) > 1:
                    breakpoint()
                return nltk.Tree(
                    t.label(), [t[0], *[anonymize_tree(c, anon='_ref_') for c in t[1:]]])
            case 'DISCARD':
                assert len(t) == 2
                c0 = anonymize_tree(t[0], anon='_ref_')
                c1 = anonymize_tree(t[1], anon='_ref_')
                return nltk.Tree(t.label(), [c0, c1])
            case 'ARITHMETIC':
                if len(t[0].split(' ')) > 1:
                    breakpoint()
                return nltk.Tree(
                    t.label(), [t[0], *[anonymize_tree(c, anon='_ref_') for c in t[1:]]])
            case 'SORT':
                assert len(t) == 2
                c0 = anonymize_tree(t[0], anon='_order_')
                c1 = anonymize_tree(t[1], anon='_ref_')
                return nltk.Tree(t.label(), [c0, c1])
            case 'COMPARISON':
                if len(t) > 3:
                    raise NotImplementedError
                if len(t[0].split(' ')) > 1:
                    breakpoint()
                c0 = t[0]
                c1 = anonymize_tree(t[1], anon='_count_')
                c1 = anonymize_tree(t[1], anon='_count_')
                return nltk.Tree(t.label(), [c0, c1, c2])
    else:
        return anon or t

def anonymize(p):
    t = make_tree(p)
    try:
        anon_tree = anonymize_tree(t)
        return anon_tree
    except Exception as e:
        return None

def tree_to_tuple(t):
    if isinstance(t, nltk.Tree):
        return (t.label(), tuple([tree_to_tuple(c) for c in t]))
    else:
        return t

def tuple_to_tree(t):
    if isinstance(t, tuple):
        return nltk.Tree(t[0], [tuple_to_tree(c) for c in t[1]])
    else:
        return t

def make_template_split(ds, seed=42):
    import random
    import numpy as np
    import pandas as pd
    import numpy.random as npr
    from collections import defaultdict
    random.seed(seed)
    npr.seed(seed)
    trndf = ds['train'].to_pandas()
    valdf = ds['validation'].to_pandas()
    df = pd.concat([trndf, valdf])
    df['tree'] = df.program.apply(make_tree)
    df['anon_tree'] = df.program.apply(anonymize)
    df['anon_tuple'] = df.anon_tree.apply(tree_to_tuple)

    template2ex = defaultdict(list)
    for idx, r in df.iterrows():
        template2ex[r.anon_tuple].append(idx)
    for k, v in template2ex.items():
        npr.shuffle(v)
    templates = df.anon_tuple.unique()
    npr.shuffle(templates)
    trn_templates = templates[:int(len(templates) * 0.8)]
    val_templates = templates[int(len(templates) * 0.8):]
    trn_exids = [idx for t in trn_templates for idx in template2ex[t][:30]]
    val_exids = [idx for t in val_templates for idx in template2ex[t][:5]]
    trnset = df.iloc[trn_exids].reset_index()
    valset = df.iloc[val_exids].reset_index()
    return trnset, valset

class Break(datasets.GeneratorBasedBuilder):
    """TODO(drop): Short description of my dataset."""

    # TODO(drop): Set up version.
    VERSION = datasets.Version("0.2.0")

    def _info(self):
        # TODO(drop): Specifies the datasets.DatasetInfo object
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            features=datasets.Features({
                'question_id': datasets.Value(dtype='string'),
                'question_text': datasets.Value(dtype='string'),
                'decomposition': datasets.Value(dtype='string'),
                'operators': datasets.Value(dtype='string'),
                'split': datasets.Value(dtype='string'),
                # 'program': datasets.Value(dtype='string'),
                # 'template': datasets.Value(dtype='string'),
            }),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # TODO(drop): Downloads the data and defines the splits
        # dl_manager is a datasets.download.DownloadManager that can be used to
        # download and extract URLs
        # dl_dir = dl_manager.download_and_extract(_URL)
        # data_dir = os.path.join(dl_dir, "drop_dataset")
        # icgs_train_split = dl_manager.download('https://github.com/inbaroren/improving-compgen-in-semparse/blob/main/data/DROP/train.json')
        # icgs_test_split = dl_manager.download('https://github.com/inbaroren/improving-compgen-in-semparse/blob/main/data/DROP/heldout_test.json')

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs=dict(split=datasets.Split.TRAIN)
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs=dict(split=datasets.Split.VALIDATION)
            ),
            datasets.SplitGenerator(
                name='template_train',
                gen_kwargs=dict(split='template_train')
            ),
            datasets.SplitGenerator(
                name='template_test',
                gen_kwargs=dict(split='template_test')
            ),
        ]

    def _generate_examples(self, split):
        """Yields examples."""
        # TODO(drop): Yields (key, example) tuples from the dataset
        if split in [datasets.Split.TRAIN, datasets.Split.VALIDATION]:
            for idx, ex in enumerate(datasets.load_dataset('break_data', name='QDMR')[split]):
                yield idx, ex
        else:
            ds = datasets.load_dataset('break_data', name='logical-forms')
            trndf, valdf = make_template_split(ds)
            df = trndf if split == 'template_train' else valdf
            df['template'] = df.anon_tuple.apply(str)
            for idx, r in df.iterrows():
                yield idx, dict(
                    question_id=r.question_id,
                    question_text=r.question_text,
                    decomposition=r.decomposition,
                    operators=r.operators,
                    split=split,
                    # program=r.program,
                    # template=r.template,
                )
