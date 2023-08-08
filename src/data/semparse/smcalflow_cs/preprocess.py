import os
import typer
import pandas as pd
from pathlib import Path

app = typer.Typer()

@app.command()
def to_tsv(k: int = 0, split: str = 'train', data_root: Path = '../data/semparse/smcalflow_cs'):
    input_file = data_root / f'reg_attn_data/data/smcalflow_cs/calflow.orgchart.event_create/source_domain_with_target_num{k}/{split}.jsonl'
    output_file = data_root / f'{k}_shot/{split}.tsv'
    df: pd.DataFrame = pd.read_json(input_file, lines=True)
    df['source'] = df['utterance']
    df['target'] = df['plan']
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df[['source', 'target']].to_csv(output_file, sep='\t', index=False, header=False)

@app.command()
def convert_all():
    for k in [0, 8, 16, 32, 64, 128]:
        for split in ['train', 'test', 'valid']:
            print(f'Converting {k} shot {split}...')
            to_tsv(k, split)

if __name__ == "__main__":
    app()