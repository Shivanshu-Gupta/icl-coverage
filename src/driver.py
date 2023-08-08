import os
import math
import torch
import numpy as np
import hydra

from datasets import Dataset
from copy import deepcopy
from functools import partial
from omegaconf import DictConfig, OmegaConf
from typer import Typer
from rich import print


from tools.utils import Logger
from tools.lm import get_enc_len_fn
from params import AllParams
from constants import Dataset as D, ExSel as ES, LLM, default_prompt_version
from prompts.few_shot import FewShotPromptTemplate2
from selector_utils import get_selector
from eval import eval, dump_prompts

# app = Typer()

def set_seeds(seed):
    import numpy as np
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # CPU random seed
    torch.cuda.manual_seed(seed)  # GPU random seed

@hydra.main(version_base=None, config_name="config")
def main(P: AllParams):
    P: AllParams = OmegaConf.to_object(P)
    if P.exp.tiny:
        P.data.n_cands, P.data.n_test = 40, 20
    print(P)
    print(P.get_output_dir())
    os.makedirs(P.get_output_dir(), exist_ok=True)
    logger = Logger(outfile=P.get_logfile())
    try:
        run_main(P, logger)
    except Exception as e:
        import traceback
        logger.log(traceback.format_exc())
        logger.log(e)

def run_main(P: AllParams, logger: Logger):
    log = logger.log
    # train_ds, test_ds, candidates, fewshot_prompt_fn, templates = get_data(P, logger)
    EP, DP, LP, SP = P.shorthand
    train_ds, candidates, test_ds = DP.get_splits(EP.data_root, 'data', EP.seed)
    templates = DP.get_templates()
    fewshot_prompt_fn = partial(FewShotPromptTemplate2,
        input_variables=templates['example_template'].input_variables,
        example_separator='\n\n', **templates)
    prompt_template = get_prompt_template(
        P, train_ds, test_ds, candidates, fewshot_prompt_fn,
        templates, logger
    )
    if P.exp.only_prompts:
        dump_prompts(
            P, test_ds, prompt_template=prompt_template,
            logger=logger, outfile=P.get_promptsfile(), debug=P.exp.debug
        )
    else:
        torch.cuda.empty_cache()
        llm = P.get_lm()
        print('Instantiating LLMChain...')
        # agent = LLMChain(prompt=prompt_template, llm=llm, verbose=P.debug)
        eval(P, test_ds, llm, prompt_template, batch_size=P.exp.batch_size,
             logger=logger, outfile=P.get_resultsfile(), debug=P.exp.debug)

def get_max_output_length(
    dataset: Dataset, example_template, llm: LLM = None, enc_len_fn = None):
    enc_len_fn = enc_len_fn or get_enc_len_fn(llm)
    test_strings = [example_template.format(**ex, test=True) for ex in dataset]
    completed_strings = [example_template.format(**ex, test=False) for ex in dataset]
    test_str_lens = [enc_len_fn(s) for s in test_strings]
    completed_str_lens = [enc_len_fn(s) for s in completed_strings]
    output_lens = [c - t for t, c in zip(test_str_lens, completed_str_lens)]
    return max(output_lens)

def get_prompt_template(
    P: AllParams, train_ds: Dataset, test_ds: Dataset, candidates: Dataset,
    fewshot_prompt_fn, templates, logger: Logger
):
    EP, DP, LP, SP = P.shorthand
    from constants import max_new_tokens_d, context_length_limit
    enc_len_fn = get_enc_len_fn(LP.lm_name)
    max_len = context_length_limit[LP.lm_name]
    subtract_gen_len = True
    if DP.dataset in max_new_tokens_d:
        max_len -= max_new_tokens_d[DP.dataset]
        subtract_gen_len = False
    else:
        max_output_len = get_max_output_length(train_ds, templates['example_template'], P.lm_name)
        max_len -= (max_output_len + 5)
        subtract_gen_len = False

    fewshot_prompt_fn = partial(fewshot_prompt_fn,
        max_len=max_len, enc_len_fn=enc_len_fn, subtract_gen_len=subtract_gen_len)

    if SP.n_shots == -1:
        P = deepcopy(P)
        SP.n_shots = 50

    if SP.selector_type == ES.RANDOM:
        fewshot_prompt = fewshot_prompt_fn(examples=list(train_ds.select(np.arange(SP.n_shots))))

    elif SP.selector_type in [
        ES.COSINE, ES.STRUCT, ES.BERTSCORE, ES.LF_COVERAGE
    ]:
        ex_selector = get_selector(P, candidates, templates['example_template'], test_ds, enc_len_fn, max_len, subtract_gen_len=subtract_gen_len)
        fewshot_prompt = fewshot_prompt_fn(example_selector=ex_selector)
    else:
        raise ValueError(f'Unknown selector_type: {SP.selector_type}')
    return fewshot_prompt

if __name__ == '__main__':
    main()