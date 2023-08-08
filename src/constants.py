from enum import Enum

class Dataset(str, Enum):
    ALPACA = 'alpaca-plus'

    AGNEWS = 'agnews'
    SST5 = 'sst5'
    RTE = 'rte'
    QNLI = 'qnli'
    MNLI = 'mnli'
    CMSQA = 'cmsqa'
    MRPC = 'mrpc'
    SST2 = 'sst2'
    DBPEDIA = 'dbpedia'
    TREC = 'trec'
    COLA = 'cola'

    ATIS = 'atis'
    GEOQUERY = 'geoquery'
    OVERNIGHT = 'overnight'
    SMCALFLOW = 'smcalflow'
    SMCALFLOW_CS = 'smcalflow-cs'
    COGS = 'cogs'
    CFQ = 'cfq'
    SPIDER = 'spider'

    BREAK = 'break'
    MTOP = 'mtop'

    BOOLQ = 'boolq'
    DROP = 'drop'

    GSM8K = 'gsm8k'
    AQUA = 'aqua'
    TABMWP = 'tabmwp'

class ExSel(str, Enum):
    RANDOM = 'random'
    BERTSCORE = 'bertscore'
    STRUCT = 'structural'
    COSINE = 'cosine'
    LF_COVERAGE = 'lf_coverage'
    EPR = 'epr'
    CEIL = 'ceil'

class LMType(str, Enum):
    OPENAI = 'openai'
    OPENAI_CHAT = 'openai_chat'
    OPT_SERVER = 'opt_server'
    HUGGINGFACE = 'huggingface'


class LLM(str, Enum):
    TEXT_DAVINCI_002 = 'text-davinci-002'
    TEXT_DAVINCI_003 = 'text-davinci-003'
    ADA = ' ada'
    CODE_DAVINCI_002 = 'code-davinci-002'
    CODE_CUSHMAN_001 = 'code-cushman-001'
    CODE_CUSHMAN_002 = 'code-cushman-002'
    GPT4 = 'gpt-4-0314'
    TURBO = 'gpt-3.5-turbo-0301'
    GPT_NEO_125M = 'gpt-neo-125M'
    OPT_13B = 'opt-13b'
    OPT_30B = 'opt-30b'
    NEO = 'EleutherAI/gpt-neo-2.7B'
    NEOX20B = 'EleutherAI/gpt-neox-20b'
    JT6B = 'togethercomputer/GPT-JT-6B-v1'
    LLAMA7B = 'llama-7B'
    LLAMA13B = 'llama-13B'
    LLAMA30B = 'llama-30B'
    STARCODER = 'bigcode/starcoder'

D = Dataset
max_new_tokens_d = {
    D.SST2: 4,
    D.AGNEWS: 4,
    D.SST5: 4,
    D.BOOLQ: 1,
    D.RTE: 2,
    D.QNLI: 1,
    D.MNLI: 1,
    D.CMSQA: 10,

    D.SMCALFLOW_CS: 256,
    D.SMCALFLOW: 200,
    D.GEOQUERY: 128,
    D.OVERNIGHT: 128,
    D.ATIS: 128,
    D.BREAK: 256,
    D.MTOP: 110,
    D.DROP: 25,

    D.GSM8K: 500,
    D.AQUA: 500,
}

context_length_limit = {
    LLM.CODE_CUSHMAN_001: 2048,
    LLM.CODE_CUSHMAN_002: 2048,
    LLM.CODE_DAVINCI_002: 8001,
    LLM.TEXT_DAVINCI_003: 4096,
    # LLM.TURBO: 4096,
    LLM.TURBO: 4000,
    LLM.GPT4: 8192,
    LLM.NEO: 2048,
    LLM.JT6B: 2048,
    LLM.NEOX20B: 2048,
    LLM.LLAMA7B: 2048,
    LLM.LLAMA13B: 2048,
    LLM.STARCODER: 7000,
}

default_prompt_version = {
    LLM.NEO: 'v2',
    LLM.JT6B: 'v2',
    LLM.NEOX20B: 'v2',
    LLM.CODE_CUSHMAN_001: 'v1',
    LLM.CODE_DAVINCI_002: 'v1',
    LLM.LLAMA7B: 'v2',
    LLM.LLAMA13B: 'v2',
    LLM.TURBO: 'v1',
    LLM.GPT4: 'v2',
    LLM.TEXT_DAVINCI_003: 'v2',
    LLM.STARCODER: 'v2',
}

lfst_prompt_cov_datasets = [D.GEOQUERY, D.OVERNIGHT]
no_prompt_cov_datasets = [D.ATIS, D.GSM8K, D.AQUA]
local_semparse_datasets = [
    D.ATIS, D.GEOQUERY, D.OVERNIGHT, D.COGS,
    D.SMCALFLOW, D.SMCALFLOW_CS,
]
semparse_datasets = [
    *local_semparse_datasets,
    D.BREAK, D.MTOP, D.CFQ, D.SPIDER, D.COGS,
]
mrc_datasets = [D.DROP]
mwp_datasets = [D.GSM8K, D.AQUA]
generation_datasets = [
    *semparse_datasets,
    *mrc_datasets,
    *mwp_datasets,
]
local_classification_datasets = [D.SST2, D.TREC, D.AGNEWS, D.RTE]
glue_datasets = [D.MNLI, D.QNLI]
superglue_datasets = [D.RTE, D.BOOLQ]
classification_datasets = [
    *local_classification_datasets,
    *glue_datasets,
    *superglue_datasets,
    D.SST5, D.CMSQA,
]

# over 1000 test instances -> pick random 1000
test_subset_datasets = [
    D.GEOQUERY, D.OVERNIGHT, D.BREAK, D.MTOP, D.DROP,
    D.BOOLQ, D.MNLI, D.QNLI, D.GSM8K, D.COGS
]
