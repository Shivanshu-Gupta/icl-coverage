from constants import LLM
from pathlib import Path


def llama_path(model_size: str = '7B'):
    if Path(f'/srv/disk01/ucinlp/shivag5/llama_hf/{model_size}').exists():
        return f'/srv/disk01/ucinlp/shivag5/llama_hf/{model_size}'
    elif Path(f'/srv/nvme0/ucinlp/shivag5/llama_hf/{model_size}/').exists():
        return f'/srv/nvme0/ucinlp/shivag5/llama_hf/{model_size}'
    else:
        raise ValueError('No llama path found')

def get_enc_len_fn(lm: LLM):
    if lm in [LLM.NEO, LLM.NEOX20B, LLM.STARCODER]:
        print(f'Using {lm} tokenizer')
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(lm.value, use_auth_token=True)
        enc_len_fn = lambda x: len(tokenizer.encode(x))
    elif lm == LLM.LLAMA7B:
        from transformers import LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained(llama_path('7B'))
        enc_len_fn = lambda x: len(tokenizer.encode(x))
    elif lm == LLM.LLAMA13B:
        from transformers import LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained(llama_path('13B'))
        enc_len_fn = lambda x: len(tokenizer.encode(x))
    elif lm == LLM.LLAMA30B:
        from transformers import LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained(llama_path('30B'))
        enc_len_fn = lambda x: len(tokenizer.encode(x))
    else:
        print(f'Using tiktoken tokenizer for {lm}')
        import tiktoken
        enc = tiktoken.encoding_for_model(lm.value)
        enc_len_fn = lambda x: len(enc.encode(x))
    return enc_len_fn
