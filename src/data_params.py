from __future__ import annotations

import attr
import numpy as np
from pathlib import Path
from datasets import load_dataset, Dataset
from prompts import (
    ExampleTemplate,
    GenerationExampleTemplate,
    ClassificationExampleTemplate,
    ContextualizedGenerationExampleTemplate,
    ContextualizedClassificationExampleTemplate,
    SemparseExampleTemplate,
)
from tools.param_impl import Parameters
from constants import Dataset as D

@attr.s(auto_attribs=True)
class DataParams(Parameters):
    dataset: D = D.GEOQUERY                         # Dataset name.
    n_test: int = -1                 # Number of test examples to use.
    n_cands: int | None = -1         # Number of candidates to select from.
    prefix: bool = True                 # Whether to use the prefix the prompt with task description.
    max_tokens: int | None = 256     # Maximum number of tokens to generate.

    task_desc: str | None = None

    split: str | None = None            # Split to use for train/test.
    test_split: str | None = None       # Split to use for test.
    train_split: str | None = None      # Split to use for train.

    prompt_version: str | None = None
    n_inputs: int = 1

    def subsample_splits(self, train_ds: Dataset, test_ds: Dataset, seed) -> tuple[Dataset, Dataset, Dataset]:
        n_cands, n_test = self.n_cands, self.n_test
        train_ds = train_ds.shuffle(seed=seed)
        candidates = train_ds.select(np.arange(n_cands)) if n_cands != -1 else train_ds
        if n_test != -1 and n_test < len(test_ds):
            test_ds = test_ds.shuffle(seed=0).select(np.arange(n_test))
        return train_ds, candidates, test_ds

    def get_dataset(self, data_root: str = '../data', dataloaders_dir: str = 'data') -> Dataset:
        raise NotImplementedError

    def get_splits(self, data_root: str = '../data', dataloaders_dir: str = 'data', seed=0):
        ds = self.get_dataset(data_root, dataloaders_dir)
        return self.subsample_splits(ds[self.train_split], ds[self.test_split], seed)


    def get_templates(self) -> dict[str, ExampleTemplate]:
        raise NotImplementedError

    def get_dataset_name(self):
        data_name_parts = [self.dataset.value]
        if hasattr(self, 'embed_context') and self.embed_context:
            data_name_parts.append('embed_context')
        return '-'.join(data_name_parts)

    def get_split_name(self):
        if self.split:
            return self.split
        elif self.train_split and self.train_split != 'train':
            return f'{self.train_split}-{self.test_split}'
        else:
            return self.test_split

    def get_prompt_name(self, lm_default_prompt_version):
        prompt_name_parts = []
        if self.prompt_version != lm_default_prompt_version:
            prompt_name_parts.append(self.prompt_version)
        if not self.prefix:
            prompt_name_parts.append('no_prefix')
        return '-'.join(prompt_name_parts)


# ---------------------------------------------------------------------------- #
#                               Semantic Parsing                               #
# ---------------------------------------------------------------------------- #

semparse_desc = 'Translate the sentence into a logical form.'
@attr.s(auto_attribs=True)
class SemanticParsing(DataParams):
    dataset: D = D.GEOQUERY
    input_feature: str = 'source'       # Name of the input feature.
    target_feature: str = 'target'      # Name of the target feature.
    n_inputs: int = 1

    # def __attrs_post_init__(self):
    #     self.n_test = self.n_test or -1

    def get_dataset(self, data_root: str = '../data', dataloaders_dir: str = 'data'):
        return load_dataset(
            name=self.dataset.value,
            path=f'{dataloaders_dir}/semparse/semparse.py',
            data_dir=f'{data_root}/semparse')

    def get_templates(self):
        task_desc = semparse_desc
        return dict(
            prefix_template=self.task_desc or task_desc if self.prefix else '',
            example_template=SemparseExampleTemplate(
                input_variables=[self.input_feature, self.target_feature],
                version=self.prompt_version))

@attr.s(auto_attribs=True)
class GeoQuery(SemanticParsing):
    dataset: D = D.GEOQUERY
    input_feature: str = 'source'       # Name of the input feature.
    target_feature: str = 'target'      # Name of the target feature.

    split: str | None = 'iid'           # Dataset split. Used for datasets in constants.datasets_with_split_arg.
    train_split: str = 'train'          # Huggingface dataset split for training.
    test_split: str = 'test'            # Huggingface dataset split for testing.
    max_tokens: int = 128

    # def __attrs_post_init__(self):
    #     self.n_test = self.n_test or 1000

    def get_splits(self, data_root: str = '../data', dataloaders_dir: str = 'data', seed=0):
        ds = self.get_dataset(data_root, dataloaders_dir)
        if self.split == 'iid':
            trn, tst = ds['standard_train'], ds['standard_test']
        else:
            trn, tst = ds[f'{self.split}_train'], ds[f'{self.split}_test']
        return self.subsample_splits(trn, tst, seed)

@attr.s(auto_attribs=True)
class SMCalFlowCS(SemanticParsing):
    dataset: D = D.SMCALFLOW_CS
    input_feature: str = 'source'       # Name of the input feature.
    target_feature: str = 'target'      # Name of the target feature.
    split: str = '32_S'                  # Dataset split. Used for datasets in constants.datasets_with_split_arg.
    max_tokens: int = 256

    def get_splits(self, data_root: str = '../data', dataloaders_dir: str = 'data', seed=0):
        from datasets import concatenate_datasets
        ds = self.get_dataset(data_root, dataloaders_dir)

        fewshot_ds = ds['fewshots'].select(np.arange(int(self.split.split('_')[0])))
        train_ds = ds['train']
        test_ds = ds['iid_test'] if 'S' in self.split else ds['comp_test']
        train_ds, candidates, test_ds = self.subsample_splits(train_ds, test_ds, seed)
        candidates = concatenate_datasets([candidates, fewshot_ds])
        return train_ds, candidates, test_ds

@attr.s(auto_attribs=True)
class Atis(SemanticParsing):
    dataset: D = D.ATIS
    input_feature: str = 'question'       # Name of the input feature.
    target_feature: str = 'logical_form'      # Name of the target feature.
    split: str = 'iid_0'              # Dataset split. Used for datasets in constants.datasets_with_split_arg.
    max_tokens: int = 128

    def get_splits(self, data_root: str = '../data', dataloaders_dir: str = 'data', seed=0):
        ds = self.get_dataset(data_root, dataloaders_dir)
        return self.subsample_splits(ds[f'{self.split}_train'], ds[f'{self.split}_test'], seed)

@attr.s(auto_attribs=True)
class Overnight(SemanticParsing):
    dataset: D = D.OVERNIGHT
    input_feature: str = 'paraphrase'       # Name of the input feature.
    target_feature: str = 'target'      # Name of the target feature.
    split: str = 'iid_0'              # Dataset split. Used for datasets in constants.datasets_with_split_arg.
    max_tokens: int = 128

    def get_dataset_name(self):
        return f'{self.dataset}-{self.input_feature}'

    def get_splits(self, data_root: str = '../data', dataloaders_dir: str = 'data', seed=0):
        ds = self.get_dataset(data_root, dataloaders_dir)
        return self.subsample_splits(ds[f'{self.split}_train'], ds[f'{self.split}_test'], seed)

@attr.s(auto_attribs=True)
class Break(DataParams):
    dataset: D = D.BREAK
    train_split: str = 'train'
    test_split: str = 'validation'
    max_tokens: int = 256

    def __attrs_post_init__(self):
        self.n_test = self.n_test or 1000

    def get_dataset(self, data_root: str = '../data', dataloaders_dir: str = 'data'):
        return load_dataset(f'{dataloaders_dir}/semparse/break/break.py')

    def get_templates(self):
        from prompts.base import BreakExampleTemplate
        task_desc = 'Decompose the sentence into a sequence of steps.'
        return dict(
            prefix_template=self.task_desc or task_desc if self.prefix else '',
            example_template=BreakExampleTemplate(version=self.prompt_version))

@attr.s(auto_attribs=True)
class MTOP(DataParams):
    dataset: D = D.MTOP
    input_feature: str = 'question'       # Name of the input feature.
    target_feature: str = 'logical_form'      # Name of the target feature.

    train_split: str = 'train'
    test_split: str = 'validation'
    max_tokens: int = 110

    # def __attrs_post_init__(self):
    #     self.n_test = self.n_test or 1000

    def get_dataset(self, data_root: str = '../data', dataloaders_dir: str = 'data'):
        return load_dataset('iohadrubin/mtop')

    def get_templates(self):
        from prompts.base import SemparseExampleTemplate
        task_desc = semparse_desc
        return dict(
            prefix_template=self.task_desc or task_desc if self.prefix else '',
            example_template=SemparseExampleTemplate(
                input_variables=[self.input_feature, self.target_feature],
                version=self.prompt_version))

@attr.s(auto_attribs=True)
class COGS(SemanticParsing):
    dataset: D = D.COGS
    input_feature: str = 'source'       # Name of the input feature.
    target_feature: str = 'target'      # Name of the target feature.
    train_split: str = 'train'
    test_split: str = 'dev'
    max_tokens: int = 128

    def __attrs_post_init__(self):
        self.n_test = self.n_test or 1000

    def get_splits(self, data_root: str = '../data', dataloaders_dir: str = 'data', seed=0):
        ds = self.get_dataset(data_root, dataloaders_dir)
        return self.subsample_splits(
            ds[self.train_split],
            ds[self.test_split],
            seed)

@attr.s(auto_attribs=True)
class QNLI(DataParams):
    dataset: D = D.QNLI
    train_split: str = 'train'
    test_split: str = 'validation'
    prompt_version: str | None = None
    embed_context: bool = False   # Whether to embed the context.
    max_tokens: int = 1
    n_input: int = 2

    # def __attrs_post_init__(self):
    #     self.n_test = self.n_test or 1000

    def get_dataset(self, data_root: str = '../data', dataloaders_dir: str = 'data'):
        return load_dataset('glue', 'qnli')

    def get_templates(self):
        from prompts.base import QNLIExampleTemplate
        task_desc = 'Answer the following questions given the passage as Yes or No:'
        return dict(
            prefix_template=self.task_desc or task_desc if self.prefix else '',
            example_template=QNLIExampleTemplate(embed_context=self.embed_context or False))

@attr.s(auto_attribs=True)
class GSM8K(DataParams):
    dataset: D = D.GSM8K
    train_split: str = 'train'
    test_split: str = 'test'
    max_tokens: int = 500
    n_input: int = 2

    def get_dataset(self, data_root: str = '../data', dataloaders_dir: str = 'data'):
        return load_dataset('gsm8k', 'main')

    def get_templates(self):
        from prompts import GSM8KExampleTemplate
        task_desc = 'Answer the following question through careful, concise step-by-step reasoning.'
        return dict(
            prefix_template=self.task_desc or task_desc if self.prefix else '',
            example_template=GSM8KExampleTemplate())

@attr.s(auto_attribs=True)
class DROP(DataParams):
    dataset: D = D.DROP
    train_split: str = 'train'
    test_split: str = 'validation'
    embed_context: bool | None = False   # Whether to embed the context.
    max_tokens: int = 25
    n_input: int = 2

    def get_dataset(self, data_root: str = '../data', dataloaders_dir: str = 'data'):
        return load_dataset(path=f'{dataloaders_dir}/drop.py')

    def get_templates(self):
        from prompts.base import MRCExampleTemplate
        task_desc = "Answer the following questions based on the passage:"
        return dict(
            prefix_template=self.task_desc or task_desc if self.prefix else '',
            example_template=MRCExampleTemplate(
                input_variables=['question', 'passage', 'answer_text'],
                embed_context=self.embed_context or False))

all_datasets = [
    GeoQuery, SMCalFlowCS, Atis, Overnight, Break, MTOP, COGS,
    QNLI,
    GSM8K, DROP,
]

ds2cls: dict[D, DataParams] = {ds_cls().dataset: ds_cls for ds_cls in all_datasets}

def test():
    data_root = Path('../data')
    dataloaders_dir = Path('data')
    for ds_cls in all_datasets[-5:]:
        print(f' {ds_cls.__name__} '.center(80, '='))
        try:
            ds = ds_cls()
            ds.prompt_version = ds.prompt_version or 'v1'
            datasets = ds.get_dataset(data_root, dataloaders_dir)
            print(datasets)
            train_ds, candidates, test_ds = ds.get_splits(data_root, dataloaders_dir)
            print(candidates)
            print(test_ds)
            print(' Templates '.center(80, '-'))
            templates = ds.get_templates()
            prefix_template = templates['prefix_template']
            example_template = templates['example_template']
            print(prefix_template)
            print(example_template.format(**candidates[0]))
        except Exception as e:
            print(f'{ds_cls.__name__} failed')
            raise e
        print()

if __name__ == '__main__':
    test()