from constants import Dataset as D, ExSel as ES, LLM

get_strings = lambda x, sep=';': x.split(sep)
get_floats = lambda x, sep=';': [float(y) for y in x.split(sep)]
get_ints = lambda x, sep=';': [int(y) for y in x.split(sep)]
get_datasets = lambda x, sep=';': [D(y) for y in x.split(sep)]
get_lms = lambda x: [LLM(y) for y in x.split(';')]
get_exsels = lambda x, sep=';': [ES(y) for y in x.split(sep)]