import re
from prompts.base import ExampleTemplate

gsm8k_prefix = "Answer the following question through careful, concise step-by-step reasoning."
class GSM8KExampleTemplate(ExampleTemplate):
    input_variables: list[str] = ['question', 'answer']
    templates: dict[str, str] = dict(
        train="Question: {source}\nSolution: {target}",
        test="Question: {source}\nSolution: ",
    )

    def format(self, test=False, embedding=False, **kwargs):
        source = self.get_source(**kwargs)
        target = self.get_target(**kwargs)
        if embedding: return source
        template = self.templates['test'] if test else self.templates['train']
        return template.format(source=source, target=target)

    def prepare_for_turbo(self, string):
        input = string[:string.find("Solution: ")].strip()
        output = string[string.find("Solution: "):].strip()
        input = input.replace("Question: ", "")
        output = output.replace("Solution: ", "")
        return input, output

    def get_source(self, **kwargs):
        return kwargs[self.input_variables[0]].strip()

    def get_target(self, **kwargs):
        return kwargs[self.input_variables[1]].strip()

    def extract_answer(self, completion):
        ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
        INVALID_ANS = "[invalid]"
        match = ANS_RE.search(completion)
        if match:
            match_str = match.group(1).strip()
            match_str = match_str.replace(",", "")
            return match_str
        else:
            # special case for turbo
            TURBO_ANS_RE = re.compile(r"Answer: \\boxed{(\-?[0-9\.\,]+)}")
            match = TURBO_ANS_RE.search(completion)
            if match:
                match_str = match.group(1).strip()
                match_str = match_str.replace(",", "")
                return match_str
            else:
                return INVALID_ANS

    def parse_output(self, lm_output: str, **kwargs):
        # based on https://github.com/openai/grade-school-math/blob/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/dataset.py#L28
        return self.extract_answer(lm_output)

    def check_output(self, pred, target, **kwargs):
        answer = self.extract_answer(target)
        is_correct = pred == answer
        return dict(accuracy=is_correct * 100,)



aqua_prefix = "Answer the following question through careful, concise step-by-step reasoning."  # taken from cot/fragments.json

class AquaExampleTemplate(ExampleTemplate):
    input_variables: list[str] = ['question', 'options', 'rationale', 'correct']
    templates: dict[str, str] = dict(
        train="Question: {source}\nChoices: {choices}\nRationale: {rationale}\nAnswer: {target}",
        test="Question: {source}\nChoices: {choices}\nRationale: ",
    )

    def format(self, test=False, embedding=False, **kwargs):
        source = self.get_source(**kwargs)
        target = self.get_target(**kwargs)
        if embedding: return source
        template = self.templates['test'] if test else self.templates['train']
        return template.format(
            source=source, target=target,
            choices=self.get_choices(**kwargs),
            rationale=self.get_rationale(**kwargs),
        )

    def get_source(self, **kwargs):
        return kwargs[self.input_variables[0]].strip()

    def get_choices(self, **kwargs):
        return ', '.join(kwargs[self.input_variables[1]])

    def get_rationale(self, **kwargs):
        return kwargs[self.input_variables[2]].strip()

    def get_target(self, **kwargs):
        return kwargs[self.input_variables[3]].strip()

    def extract_answer(self, completion):
        ANS_RE = re.compile(r"Answer: ([A-E])")
        INVALID_ANS = "[invalid]"
        match = ANS_RE.search(completion)
        if match:
            match_str = match.group(1).strip()
            return match_str
        else:
            return INVALID_ANS

    def parse_output(self, lm_output: str, **kwargs):
        # based on https://github.com/openai/grade-school-math/blob/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/dataset.py#L28
        return self.extract_answer(lm_output)

    # def check_output(self, lm_output, **kwargs):
    #     prediction = self.extract_answer(lm_output)
    #     answer = self.get_target(**kwargs)
    #     return prediction == answer

    def check_output(self, pred, target, **kwargs):
        is_correct = pred == target
        return dict(accuracy=is_correct * 100,)
