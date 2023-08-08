import time
import openai
from functools import lru_cache, cache
from itertools import cycle
from more_itertools import chunked


def get_default_api_key_pool():
    print("Loading API keys from ../../openai_keys.txt")
    return [l.split(' ')[0].strip() for l in open("../../openai_keys.txt").readlines()]

class OpenAIEngine:
    def __init__(self, id, api_key, engine, temperature, max_tokens, top_p,
                 frequency_penalty, presence_penalty):
        print(f'Initializing engine {id} with API key {api_key}')
        self.id = id
        self.api_key = api_key
        self.engine = engine
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.next_slot = time.time()
        # rate_limit_per_minute = 20
        # self.delay = 60.0 / rate_limit_per_minute
        self.delay = 6

    def __call__(self, prompts, stop=["\n"]):
        self.next_slot = time.time() + self.delay
        try:
            openai.api_key = self.api_key
            outputs = [""] * len(prompts)
            response = openai.Completion.create(engine=self.engine,
                                                prompt=prompts,
                                                temperature=self.temperature,
                                                max_tokens=self.max_tokens,
                                                top_p=self.top_p,
                                                frequency_penalty=self.frequency_penalty,
                                                presence_penalty=self.presence_penalty,
                                                stop=stop)
            # output = response["choices"][0]["text"].strip()
            for choice in response.choices:
                outputs[choice.index] = choice.text.strip()
            print(f'Engine {self.id} completed {len(prompts)} prompts')
            return outputs
        except (openai.error.RateLimitError, openai.error.ServiceUnavailableError) as e:
            print(f'Engine {self.id} failed: {e}')
            raise e
        # Raise exceptions for any errors not specified
        except Exception as e:
            raise e

class OpenAIEnginePool:
    def __init__(self, *engine_args, batch_size=5, api_keys=None):
        self.engine_args = engine_args
        self.batch_size = batch_size
        api_keys = api_keys or get_default_api_key_pool()
        # api_keys = cycle(api_keys)
        self.engines = [OpenAIEngine(id, api_key, *engine_args)
                        for id, api_key in enumerate(api_keys)]
        self.cache = {}

    def __call__(self, prompts, stop=["\n"]):
        # from tools.track import get_progress_bar
        # with get_progress_bar() as bar:
            # task = bar.add_task("Completing prompts", total=len(prompts))
        prompts_to_complete = [p for p in prompts if p not in self.cache]
        completed = len(prompts) - len(prompts_to_complete)
        # print(f'Completions for {completed}/{len(prompts)} prompts found in cache')
        # bar.update(task, completed=completed)
        print(f'Completing {len(prompts_to_complete)}/{len(prompts)} prompts in batches of {self.batch_size}')
        for batch in chunked(prompts_to_complete, self.batch_size):
            _outputs = self.complete_batch(batch, stop=stop)
            for prompt, output in zip(batch, _outputs):
                self.cache[prompt] = output
            completed += len(_outputs)
            # print(f'Completed {len(outputs)}/{len(prompts)} prompts')
            # bar.update(task, completed=completed)
        outputs = [self.cache[p] for p in prompts]
        return outputs

    def complete_batch(self, prompts, stop=["\n"]):
        # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_handle_rate_limits.ipynb
        while True:
            try:
                engine = min(self.engines, key=lambda e: e.next_slot)
                print(f'Using engine {engine.id} in {engine.next_slot - time.time():.2f} seconds')
                slot = engine.next_slot
                import pause
                pause.until(slot)
                outputs = engine(prompts, stop=stop)
                return outputs
            except (openai.error.RateLimitError, openai.error.ServiceUnavailableError) as e:
                continue
            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e



@cache
def get_pipeline(model, max_new_tokens, temperature, top_p, gpu):
    import torch
    from transformers import pipeline
    device = torch.device("cuda:" + gpu if torch.cuda.is_available() else "cpu")
    pipe = pipeline(model=model, max_new_tokens=max_new_tokens,
                    temperature=max(temperature, 1e-3), top_p=top_p, do_sample=False,
                    # device=device,
                    device_map='auto',
                    )
    pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id
    pipe.tokenizer.padding_side = 'left'
    return pipe

def get_gptjt6B_output(prompt, args):
    pipe = get_pipeline("togethercomputer/GPT-JT-6B-v1", args.max_tokens, args.temperature, args.top_p, '1')
    if not isinstance(prompt, list): return pipe(prompt)[0]['generated_text']
    else: return [output[0]['generated_text'] for output in pipe(prompt, batch_size=10)]

if False:
    @lru_cache(maxsize=10000)
    def call_openai(engine, prompts, temperature, max_tokens, top_p, frequency_penalty, presence_penalty, api_keys=None):
        key_pool = get_default_api_key_pool() if api_keys is None else cycle(api_keys)
        while True:
            try:
                k, key = next(key_pool)
                openai.api_key = key
                outputs = [""] * len(prompts)
                response = openai.Completion.create(engine=engine,
                                                    prompt=prompts,
                                                    temperature=temperature,
                                                    max_tokens=max_tokens,
                                                    top_p=top_p,
                                                    frequency_penalty=frequency_penalty,
                                                    presence_penalty=presence_penalty,
                                                    stop=["\n"])
                # output = response["choices"][0]["text"].strip()
                for choice in response.choices:
                    outputs[choice.index] = choice.text.strip()
                break
            except openai.error.RateLimitError as e:
                print(f'key {k}: {e}')
            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e
        return outputs

    @lru_cache(maxsize=10000)
    def call_gpt3(engine, prompt, temperature, max_tokens, top_p, frequency_penalty, presence_penalty):
        patience = 100
        while True:
            try:
                response = openai.Completion.create(engine=engine,
                                                    prompt=prompt,
                                                    temperature=temperature,
                                                    max_tokens=max_tokens,
                                                    top_p=top_p,
                                                    frequency_penalty=frequency_penalty,
                                                    presence_penalty=presence_penalty,
                                                    stop=["\n"])
                output = response["choices"][0]["text"].strip()
                break
            except Exception as e:
                print(e)
                patience -= 1
                if not patience:
                    print("!!! running out of patience waiting for OpenAI")
                else:
                    time.sleep(0.1)
        return output
