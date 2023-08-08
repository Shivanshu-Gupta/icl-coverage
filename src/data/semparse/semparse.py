import os
import json
import pandas as pd
import datasets
from pathlib import Path

logger = datasets.logging.get_logger(__name__)

class Semparse(datasets.GeneratorBasedBuilder):
    def _info(self):
        features = {
            'qid': datasets.Value('string'),
            'source': datasets.Value('string'),
            'target': datasets.Value('string'),
        }
        if self.config.name == 'overnight':
            features['paraphrase'] = datasets.Value('string')
            features['template'] = datasets.Value('string')
        elif self.config.name == 'smcalflow-cs':
            features['original_target'] = datasets.Value('string')
            # features['paraphrase'] = datasets.Value('string')
        elif self.config.name == 'smcalflow':
            features['template'] = datasets.Value('string')
        return datasets.DatasetInfo(
            description=f'{self.config.name} dataset',
            features=datasets.Features(features),
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        ds_name = self.config.name
        def get_common_split(ds_name, split):
            return datasets.SplitGenerator(
                name=split,
                gen_kwargs={
                    'filepath': f'{ds_name}.{split}.jsonl',
                    'split': split,
                }
            )
        if ds_name == 'geoquery':
            def get_geoquery_split(split_name, split):
                return datasets.SplitGenerator(
                    name=f'{split_name.replace("/", "_")}_{split}',
                    gen_kwargs={
                        'filepath': f'geoquery.all.jsonl',
                        'splitfile': f'{split_name}.json',
                        'split': split
                    }
                )
            return [
                get_geoquery_split('csl/length', 'train'),
                get_geoquery_split('csl/length', 'test'),
                *[get_geoquery_split(f'csl/template_{i}', split) for i in range(1, 4) for split in ['train', 'test']],
                *[get_geoquery_split(f'csl/tmcd_{i}', split) for i in range(1, 4) for split in ['train', 'test']],
                *[get_geoquery_split(f'{split_type}', split) for split_type in ['standard', 'length_1', 'template_1', 'tmcd_1'] for split in ['train', 'test']],
                # get_common_split(ds_name, datasets.Split.TRAIN),
                # get_common_split(ds_name, datasets.Split.TEST),
            ]
        elif ds_name == 'atis':
            def get_overnight_split(split_name, split):
                return datasets.SplitGenerator(
                        name=f'{split_name}_{split}',
                        gen_kwargs={'filepath': f'{split_name}/atis.{split}.jsonl',}
                    )
            return [
                get_overnight_split('iid_0', 'train'),
                get_overnight_split('iid_0', 'test'),
                get_overnight_split('iid_1', 'train'),
                get_overnight_split('iid_1', 'test'),
                get_overnight_split('template_0', 'train'),
                get_overnight_split('template_0', 'test'),
                get_overnight_split('template_1', 'train'),
                get_overnight_split('template_1', 'test'),
            ]
        elif ds_name == 'overnight':
            def get_overnight_split(split_name, split):
                return datasets.SplitGenerator(
                        name=f'socialnetwork_{split_name}_{split}',
                        gen_kwargs={'filepath': f'socialnetwork/{split_name}/{split}.jsonl',}
                    )
            return [
                get_overnight_split('iid_0', 'train'),
                get_overnight_split('iid_0', 'test'),
                get_overnight_split('template_0', 'train'),
                get_overnight_split('template_0', 'test'),
                # get_overnight_split('subtree_0', 'train'),
                # get_overnight_split('subtree_0', 'test'),
            ]
        elif ds_name == 'smcalflow':
            def get_overnight_split(split_name, split):
                return datasets.SplitGenerator(
                        name=f'{split_name}_{split}',
                        gen_kwargs={'filepath': f'splits/{split_name}/{split}.jsonl',}
                    )
            return [
                get_overnight_split('iid_0', 'train'),
                get_overnight_split('iid_0', 'test'),
                get_overnight_split('template_0', 'train'),
                get_overnight_split('template_0', 'test'),
                # get_overnight_split('subtree_0', 'train'),
                # get_overnight_split('subtree_0', 'test'),
            ]

        elif ds_name == 'cogs':
            def get_cogs_split(split):
                return datasets.SplitGenerator(
                        name=split,
                        gen_kwargs={
                            'filepath': f'{split}.tsv',
                            'split': split,
                        }
                    )
            return [get_cogs_split(split) for split in ['train', 'dev', 'test', 'gen', 'train_100']]
        elif ds_name == 'smcalflow-cs':
            def get_smcalflow_split(split):
                file = split
                if 'iid' in split or 'comp' in split:
                    file = split.split('_')[1]
                return datasets.SplitGenerator(
                        name=split,
                        gen_kwargs={
                            # 'filepath': f'{split}.simplified.tsv',
                            'filepath': f'{file}.simplified.jsonl',
                            # 'filepath': f'{split}.simplified.paraphrased.jsonl',
                            'split': split,
                        }
                    )
            return [get_smcalflow_split(split) for split in [
                'train', 'fewshots',
                'valid', 'test',
                'iid_valid', 'iid_test',
                'comp_valid', 'comp_test']]
        else:
            return [
                get_common_split(ds_name, datasets.Split.TRAIN),
                get_common_split(ds_name, datasets.Split.TEST),
            ]

    def _generate_examples(self, filepath, splitfile=None, split=None):
        filepath = Path(self.config.data_dir) / self.config.name / filepath
        print(f'generating {self.config.name} examples from {filepath}')
        if self.config.name == 'smcalflow-simplified':
            idx = 0
            for i, line in enumerate(open(filepath)):
                ex = json.loads(line)
                for j, turn in enumerate(ex['turns']):
                    source  = turn['user_utterance']['original_text']
                    target = turn['lispress']
                    if target.count("(") <= 2: continue
                    yield idx, {
                        # 'qid': f'smcalflow_{i}_{j}',
                        'qid': f'smcalflow_{split}_{idx}',
                        'source': source.strip(),
                        'target': target.strip(),
                    }
                    idx += 1
        elif self.config.name == 'cogs':
            df = pd.read_csv(filepath, sep='\t', names=['source', 'target'])
            for i, row in df.iterrows():
                yield i, row.to_dict() | dict(qid=f'cogs_{split}_{i}')
        elif False and self.config.name == 'smcalflow-cs':
            df = pd.read_csv(filepath, sep='\t', names=['source', 'target'])
            for i, row in df.iterrows():
                yield i, row.to_dict() | dict(qid=f'smcalflows-cs_{split}_{i}')
        else:
            dataset = self.config.name
            df = pd.read_json(filepath, orient='records', lines=True)
            if dataset == 'smcalflow-cs':
                if split == 'iid_valid':
                    df = df.iloc[662:]
                elif split == 'comp_valid':
                    df = df.iloc[:662]
                elif split == 'iid_test':
                    df = df.iloc[663:]
                elif split == 'comp_test':
                    df = df.iloc[:663]

            if splitfile is not None:
                if 'qid' not in df.columns:
                    df['qid'] = [f'{dataset}_{i}' for i in range(len(df))]
                splitfile = os.path.join(self.config.data_dir, dataset, 'splits', splitfile)
                idxs = json.load(open(splitfile))[split]
                df = df.iloc[idxs]

            if dataset == 'smcalflow-cs':  # drop instances with empty targets
                df = df.dropna(subset=['target'])
            elif dataset == 'smcalflow':   # drop instances with very long targets
                import tiktoken
                enc = tiktoken.get_encoding("gpt2")
                idxs = [i for i, row in df.iterrows()
                        if len(enc.encode(row['target'])) <= 180]
                df = df.iloc[idxs]

            for i, row in df.iterrows():
                ex = {
                    'qid': row.get('qid', f'{dataset}_{split}_{i}'),
                    'source': row['source'],
                    'target': row['target'],
                }
                # breakpoint()
                if dataset == 'overnight':
                    ex['paraphrase'] = row['paraphrased']
                    ex['template'] = row['anonymized_target']
                elif dataset == 'smcalflow-cs':
                    ex |= row
                    # ex['original_target'] = row['original_target']
                    # ex['paraphrase'] = row['paraphrase']
                elif dataset == 'smcalflow':
                    ex['template'] = row['template']
                yield i, ex
