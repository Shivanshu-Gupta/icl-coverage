import sys
import ast
import pandas as pd
import nltk
from collections import defaultdict
from itertools import product
from functools import cache
sys.path.append('../../..')
from tools.track import track
from rich import print
from datasets import load_dataset
from collections import Counter

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

def ops_count(t, op2count):
    if isinstance(t, nltk.Tree):
        op2count[t.label()] += 1
        for c in t:
            ops_count(c, op2count)

def ops_childcount(t, op2childcount):
    if isinstance(t, nltk.Tree):
        op2childcount[t.label()].add(len(t))
        for c in t:
            ops_childcount(c, op2childcount)

def ops_idxs(idx, t, op2idxs):
    if isinstance(t, nltk.Tree):
        op2idxs[t.label()].append(idx)
        for c in t:
            ops_idxs(idx, c, op2idxs)

def label_counts(t, label2count):
    if isinstance(t, nltk.Tree):
        label2count[t.label()] += 1
        for c in t:
            label_counts(c, label2count)
    else:
        label2count[t] += 1

if __name__ == '__main__':
    ds = load_dataset('break_data', name='logical-forms')
    df = ds['train'].to_pandas()
    df['tree'] = df.program.apply(make_tree)
    op2count = defaultdict(int)
    for t in df.tree:
        if t is not None:
            ops_count(t, op2count)

    op2childcount = defaultdict(set)
    for t in df.tree:
        if t is not None:
            ops_childcount(t, op2childcount)

    op2idxs = defaultdict(list)
    for idx, t in enumerate(df.tree):
        if t is not None:
            ops_idxs(idx, t, op2idxs)

    label2count = Counter()
    for t in df.tree:
        if t is not None:
            label_counts(t, label2count)

    {'FILTER': 41236,
    'SELECT': 72450,
    'AGGREGATE': 19531,
    'PROJECT': 50996,
    'GROUP': 4980,
    'INTERSECTION': 1297,
    'UNION': 2710,
    'COMPARATIVE': 9937,
    'SUPERLATIVE': 1579,
    'BOOLEAN': 15695,
    'DISCARD': 1696,
    'ARITHMETIC': 2407,
    'SORT': 268,
    'COMPARISON': 1554}


    df['anon_tree'] = df.program.apply(anonymize)
    df['anon_tuple'] = df.anon_tree.apply(tree_to_tuple)
    anon2count = Counter(df.anon_tuple)
    import random
    import numpy as np
    import numpy.random as npr
    random.seed(42)
    npr.seed(42)
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
    trnset = df.iloc[trn_exids]
    valset = df.iloc[val_exids]

    sorted_counts = [x[1] for x in anon2count.most_common()]

    [('SELECT', 54491),
    ('_ent_', 54491),
    ('_rel_', 49819),
    ('PROJECT', 49514),
    ('_ref_', 33223),
    ('FILTER', 26534),
    ('AGGREGATE', 9465),
    ('count', 6548),
    ('COMPARATIVE', 6215),
    ('GROUP', 4566),
    ('UNION', 2593),
    ('ARITHMETIC', 2263),
    ('max', 2054),
    ('difference', 1930),
    ('SUPERLATIVE', 1544),
    ('min', 1351),
    ('DISCARD', 1313),
    ('sum', 1038),
    ('INTERSECTION', 1016),
    ('avg', 336),
    ('SORT', 266),
    ('BOOLEAN', 100),
    ('if_exist', 96),
    ('orders', 18),
    ('of', 14),
    ('division', 13),
    ('visits', 12),
    ('salary', 12),
    ('students', 10),
    ('precipitation', 8),
    ('policies', 8),
    ('problems', 8),
    ('members', 8),
    ('passengers', 6),
    ('addressed', 6),
    ('employees', 6),
    ('age', 6),
    ('present', 4),
    ('points', 4),
    ('logical_and', 4),
    ('bathrooms', 4),
    ('attendees', 4),
    ('claims', 4),
    ('speed', 4),
    ('followers', 4),
    ('win', 4),
    ('patients', 4),
    ('players', 4),
    ('amenities', 4),
    ('transactions', 4),
    ('scientists', 4),
    ('customers', 4),
    ('_order_', 3),
    ('citations', 2),
    ('multiplication', 2),
    ('appear', 2),
    ('sides', 2),
    ('mountains', 2),
    ('amounts', 2),
    ('reservations', 2),
    ('deliveries', 2),
    ('addresses', 2),
    ('balance', 2),
    ('competitions', 2),
    ('factories', 2),
    ('documents', 2),
    ('tracks', 2),
    ('platforms', 2),
    ('stations', 2),
    ('cards', 2),
    ('hours', 2),
    ('amount', 2)]

