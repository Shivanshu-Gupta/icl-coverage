# Lint as: python3
import os
import re
import json

import datasets
from datasets.tasks import QuestionAnsweringExtractive


logger = datasets.logging.get_logger(__name__)

def normalize_answer(text, unit):
    # ["1,000", "123", "3/4", "56.456", "$56.4", "-3", "-10.02", "-3/2"]

    text = re.sub("^[\$]", "", text)
    text = re.sub("[\,\.\,\/]$", "", text)
    result = re.match("^[-+]?[\d,./]+$", text)

    if result is not None:
        # is number?
        text = text.replace(",", "")
        result = re.match("[-+]?\d+$", text)
        try:
            if result is not None:
                number = int(text)
            elif "/" in text:
                nums = text.split("/")
                number = round(float(nums[0]) / float(nums[1]), 3)
            else:
                number = round(float(text), 3)
            number = str(number)
            number = re.sub(r"\.[0]+$", "", number)
            return number
        except:
            return text
    else:
        # is text
        if unit:
            text = text.replace(unit, "").strip()
        return text

class TabMWPConfig(datasets.BuilderConfig):
    """BuilderConfig for TabMWP."""

    def __init__(self, **kwargs):
        """BuilderConfig for TabMWP.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(TabMWPConfig, self).__init__(**kwargs)


class TabMWP(datasets.GeneratorBasedBuilder):
    """TabMWP: The Stanford Question Answering Dataset. Version 1.1."""

    BUILDER_CONFIGS = [
        TabMWPConfig(
            name="plain_text",
            version=datasets.Version("1.0.0", ""),
            description="Plain text",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description="",
            features=datasets.Features(
                {
                    "pid": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "choices": datasets.Sequence(datasets.Value("string")),
                    "answer": datasets.Value("string"),
                    "target": datasets.Value("string"),
                    "unit": datasets.Value("string"),
                    "table_title": datasets.Value("string"),
                    "table": datasets.Value("string"),
                    "row_num": datasets.Value("int32"),
                    "column_num": datasets.Value("int32"),
                    "solution": datasets.Value("string"),
                    "ques_type": datasets.Value("string"),
                    "ans_type": datasets.Value("string"),
                    "grade": datasets.Value("int32"),
                    "split": datasets.Value("string"),
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
            homepage="https://rajpurkar.github.io/TabMWP-explorer/",
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": 'problems_train.json'}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": 'problems_test.json'}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": 'problems_dev.json'}),
            datasets.SplitGenerator(name='test1k', gen_kwargs={"filepath": 'problems_test1k.json'}),
            datasets.SplitGenerator(name='dev', gen_kwargs={"filepath": 'problems_dev1k.json'}),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepath)
        problems = json.load(open(os.path.join(self.config.data_dir, filepath)))
        for pid, prob in problems.items():
            prob['pid'] = pid
            prob['target'] = normalize_answer(prob['answer'], prob['unit'])
            del prob['table_for_pd']
            yield int(pid), prob
