# Datasets

```bash
mkdir third_party
git clone git@github.com:google-research/language.git third_party/language
```

## GeoQuery

```bash
mkdir -p data/semparse/geoquery/
wget http://www.cs.utexas.edu/\~ml/wasp/geo-funql/corpus.xml -P data/semparse/geoquery/raw
wget ftp://ftp.cs.utexas.edu/pub/mooney/nl-ilp-data/geosystem/geobase -P data/semparse/geoquery/raw
(cd third_party/language; python -m language.compgen.nqg.tasks.geoquery.write_dataset --corpus ../../data/semparse/geoquery/raw/corpus.xml --geobase ../../data/semparse/geoquery/raw/geobase --output ../../data/semparse/geoquery/geoquery.tsv)
```

## COGS

```bash
mkdir -p data/semparse/cogs
git clone git@github.com:najoungkim/COGS data/semparse/cogs/COGS
cd third_party/language
for split in train test gen dev train_100; do
  python -m language.compgen.csl.tasks.cogs.tools.preprocess_cogs_data --input ../../data/semparse/cogs/COGS/data/${split}.tsv --output ../../data/semparse/cogs/cogs.${split}.tsv
done
```

## SMCalFlow-CS

```bash
DATA_DIR=data/semparse/smcalflow-cs
mkdir -p $DATA_DIR
wget https://www.cs.cmu.edu/\~pengchey/reg_attn_data.zip -P $DATA_DIR
unzip $DATA_DIR/reg_attn_data.zip -d $DATA_DIR
for split in train valid test; do
  python src/data/semparse/smcalflow_cs/preprocess.py to-jsonl --data-root data/semparse/smcalflow_cs --k 0 --split $split --data-root $DATA_DIR
  python src/data/semparse/smcalflow_cs/preprocess.py to-jsonl --data-root data/semparse/smcalflow_cs --k 128 --split $split --data-root $DATA_DIR
done
cp $DATA_DIR/0_shot/* $DATA_DIR
tail -n128 $DATA_DIR/128_shot/train.jsonl > $DATA_DIR/fewshots.jsonl

OPENDF_DIR=third_party/OpenDF
git clone https://github.com/telepathylabsai/OpenDF $OPENDF_DIR
# use the modified version of OpenDF
for split in train fewshots valid test; do
  PYTHONPATH=$(pwd)/$OPENDF_DIR python $OPENDF_DIR/opendf/dialog_simplify.py $DATA_DIR/${split}.jsonl $DATA_DIR/${split}.simplified.jsonl
done
```

## Overnight

Splits copied from structural-diversity repo.

## ATIS

Splits copied from structural-diversity repo.