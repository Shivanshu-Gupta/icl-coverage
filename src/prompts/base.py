import attr
import typing
import evaluate

common_templates = {
    'v1': {
        "train": """
{input_label}: {source}
{output_label}: {target}""".lstrip(),
        'test': """
{input_label}: {source}
{output_label}: """.lstrip()
    },

    'v2': {
        'train': "{source}\t{target}",
        'test': "{source}\t"
    },

    'with_context': {
        'train': """
{context_label}: {context}
{input_label}: {source}
{output_label}: {target}""".lstrip(),
        'test': """
{context_label}: {context}
{input_label}: {source}
{output_label}: """.lstrip()
    }
}

class ExampleTemplate:
    input_variables: list[str]

    def get_source(self, **kwargs):
        raise NotImplementedError

    def get_target(self, **kwargs):
        return NotImplementedError

    def format(self, test=False, embedding=False, **kwargs):
        raise NotImplementedError

    def parse_output(self, lm_output: str, **kwargs) -> str:
        raise lm_output.strip()

    def check_output(self, pred, actual_target, **kwargs):
        is_correct = pred == actual_target
        return dict(accuracy=is_correct * 100,)

    def undo_format(self, string):
        raise NotImplementedError


@attr.s(auto_attribs=True)
class GenerationExampleTemplate(ExampleTemplate):
    input_variables: list[str] = ['input', 'output']
    input_label: str = 'Input'
    output_label: str = 'Output'
    version: str = 'v1'

    def get_source(self, **kwargs):
        return kwargs[self.input_variables[0]].strip()

    def get_target(self, **kwargs):
        return kwargs[self.input_variables[1]].strip()

    @property
    def templates(self) -> dict[str, str]:
        return common_templates[self.version]

    def format(self, test=False, embedding=False, **kwargs):
        source = self.get_source(**kwargs)
        target = self.get_target(**kwargs)
        if embedding: return source
        template = self.templates['test'] if test else self.templates['train']
        return template.format(
            source=source, target=target,
            input_label=self.input_label,
            output_label=self.output_label)

    def parse_output(self, lm_output: str, **kwargs) -> str:
        if self.version == 'v2':
            return lm_output
        elif self.version == 'v1':
            if lm_output.startswith(f'{self.output_label}: '):
                # output label needs to be stripped
                # eg. with chat LMs like TURBO
                return lm_output[len(f'{self.output_label}: '):].strip()
            else:
                return lm_output
        else:
            raise NotImplementedError

    def undo_format(self, string):
        # NOTE: this would fail if the source/target themselves contain the separatorS
        if self.version == 'v2':
            source = string[:string.find('\t')]
            target = string[string.find('\t') + 1:]
        else:
            source = string[:string.find('\n')]
            target = string[string.find('\n') + 1:]
            source = source[len(f'{self.input_label}: '):]
            target = target[len(f'{self.output_label}: '):]
            # source = source[source.find(': ') + len(': '):]
            # target = target[target.find(': ') + len(': '):]
        return dict(source=source, target=target)

@attr.s(auto_attribs=True)
class ClassificationExampleTemplate(GenerationExampleTemplate):
    choices: list[str] | dict[typing.Any, str]
    input_variables: list[str] = ['text', 'label']
    input_label: str = 'Text'
    output_label: str = 'Label'
    version: str = 'v1'

    def get_target(self, **kwargs):
        """ target for the LM to predict """
        label = kwargs[self.input_variables[-1]]
        if isinstance(self.choices, list) and isinstance(label, int):
            label = self.choices[label]
        elif isinstance(self.choices, dict) and label in self.choices:
            label = self.choices[label]
        return label

    def get_choices(self, **kwargs):
        """
        target candidates
        - classes in case of classification
        - choices in case of MCQ questions
        """
        if isinstance(self.choices, list):
            return self.choices
        elif isinstance(self.choices, dict):
            return list(self.choices.values())
        else:
            raise NotImplementedError

    def check_output(self, pred, **kwargs):
        target = self.get_target(**kwargs)
        is_correct = pred == target
        return dict(accuracy=is_correct * 100,)

class ContextMixin:
    input_variables: list[str] = ['input', 'context', 'output']
    context_label: str = 'Context'
    version: str = 'with_context'
    embed_context: bool = False

    def get_context(self, **kwargs):
        return kwargs[self.input_variables[1]].strip()

    @property
    def template(self):
        assert self.version == 'with_context', 'ContextualizedClassificationExampleTemplate only works with prompt version="with_context"'
        return common_templates[self.version]

    def get_embedding_text(self, source, context):
        return source if not self.embed_context else f'{context}\n{source}'

    def format(self, test=False, embedding=False, **kwargs):
        source = self.get_source(**kwargs)
        context = self.get_context(**kwargs)
        target = self.get_target(**kwargs)
        if embedding:
            return self.get_embedding_text(source, context)
        template = self.templates['test'] if test else self.templates['train']
        return template.format(
            source=source, context=context, target=target,
            input_label=self.input_label,
            context_label=self.context_label,
            output_label=self.output_label
        )

@attr.s(auto_attribs=True)
class ContextualizedGenerationExampleTemplate(ContextMixin, GenerationExampleTemplate):
    input_variables: list[str] = ['input', 'context', 'output']
    context_label: str = 'Context'
    version: str = 'with_context'
    embed_context: bool = False

@attr.s(auto_attribs=True)
class ContextualizedClassificationExampleTemplate(ContextMixin, ClassificationExampleTemplate):
    input_variables: list[str] = ['text', 'context', 'label']
    context_label: str = 'Context'
    version: str = 'with_context'
    embed_context: bool = False

semparse_prefix = "Translate the sentence into a logical form."
@attr.s(auto_attribs=True)
class SemparseExampleTemplate(GenerationExampleTemplate):
    input_variables: list[str] = ['source', 'target']
    input_label: str = 'Sentence'
    output_label: str = 'Logical Form'

break_prefix = "Decompose the sentence into a sequence of steps."
class BreakEvaluator():
    def __init__(self):
        import sys
        sys.path.append("third_party/qdecomp_with_dependency_graphs")
        from dependencies_graph.evaluation.logical_form_matcher import LogicalFromStructuralMatcher
        from dependencies_graph.evaluation.qdmr_to_logical_form_tokens import \
            QDMRToQDMRStepTokensConverter
        from evaluation.normal_form.normalized_graph_matcher import \
            NormalizedGraphMatchScorer
        from scripts.eval.evaluate_predictions import format_qdmr
        self.converter = QDMRToQDMRStepTokensConverter()
        self.matcher = LogicalFromStructuralMatcher()
        self.scorer = NormalizedGraphMatchScorer()
        self.format_qdmr = format_qdmr
        sys.path.pop()
    def lfem(self, x):
        try:
            question, generated, decomposition, index = x['question_text'], x['pred'], x['actual_target'], x['question_id']
            gold = self.format_qdmr(decomposition)
            pred = self.format_qdmr(generated)
            decomp_lf = self.converter.convert(question_id=str(index), question_text=question,
                                                decomposition=pred.to_break_standard_string())
            gold_lf = self.converter.convert(question_id=str(index), question_text=question,
                                                decomposition=gold.to_break_standard_string())
            s = self.matcher.is_match(question_id=str(index), question_text=question, graph1=decomp_lf,
                                        graph2=gold_lf)
            return s
        except Exception as e:
            return False

@attr.s(auto_attribs=True)
class BreakExampleTemplate(GenerationExampleTemplate):
    input_variables: list[str] = ['question_text', 'decomposition']
    input_label: str = 'Sentence'
    output_label: str = 'Decomposition'
    evaluator: BreakEvaluator = attr.field(factory=BreakEvaluator)

    def get_target(self, **kwargs):
        return kwargs[self.input_variables[1]].strip().replace('  ', ' ')

    def check_output(self, pred, actual_target, **kwargs):
        is_correct = pred == actual_target
        lfem = self.evaluator.lfem(kwargs | dict(pred=pred, actual_target=actual_target))
        return dict(accuracy=is_correct * 100, lfem=lfem * 100)

@attr.s(auto_attribs=True)
class NL2BashExampleTemplate(GenerationExampleTemplate):
    input_variables: list[str] = ['nl', 'bash']
    input_label: str = 'Sentence'
    output_label: str = 'Bash'
    bleu: typing.Any = attr.field(factory=lambda: evaluate.load('bleu'))

    def check_output(self, pred, actual_target, **kwargs):
        is_correct = pred == actual_target
        bleu = self.bleu.compute(predictions=[list(pred)], references=[[list(target)]])['bleu']
        return dict(accuracy=is_correct * 100, bleu=bleu * 100)


@attr.s(auto_attribs=True)
class MRCExampleTemplate(ContextualizedGenerationExampleTemplate):
    input_variables: list[str] = ['question', 'passage', 'answer']
    context_label: str = 'Question'
    input_label: str = 'Passage'
    output_label: str = 'Answer'

class DropExampleTemplate(MRCExampleTemplate):
    input_variables: list[str] = ['passage', 'question', 'answer_text']

@attr.s(auto_attribs=True)
class RTEExampleTemplate(ContextualizedClassificationExampleTemplate):
    choices: list[str] = ["True", "False"]
    input_variables: list[str] = ['hypothesis', 'premise', 'label']
    input_label: None = None
    context_label: None = None
    output_label: None = None

    @property
    def templates(self) -> dict[str, str]:
        return dict(
            train="""
{context}
Question: {source} True or False?
Answer: {target}""".lstrip(),
            test="""
{context}
Question: {source} True or False?
Answer: """.lstrip()
    )

@attr.s(auto_attribs=True)
class QNLIExampleTemplate(ContextualizedClassificationExampleTemplate):
    choices: list[str] = ["Yes", "No"]
    input_variables: list[str] = ['question', 'sentence', 'label']
    input_label: None = None
    context_label: None = None
    output_label: None = None

    @property
    def templates(self) -> dict[str, str]:
        return dict(
            train='{context} Can we know "{source}"? {target}',
            test='{context} Can we know "{source}"? ',
            embed='{context} Can we know "{source}"?',
        )

    def get_embedding_text(self, source, context):
        # return source if not self.embed_context else self.templates['embed'].format(source=source, context=context)
        return self.templates['embed'].format(source=source, context=context)

    def parse_output(self, lm_output: str, **kwargs) -> str:
        return lm_output.strip()

    def undo_format(self, string):
        raise NotImplementedError

@attr.s(auto_attribs=True)
class MNLIExampleTemplate(QNLIExampleTemplate):
    choices: list[str] = ["Yes", "Maybe", "No"]
    input_variables: list[str] = ['hypothesis', 'premise', 'label']

    @property
    def templates(self) -> dict[str, str]:
        return dict(
            train='{context} Can we say "{source}"? {target}',
            test='{context} Can we say "{source}"? ',
            embed='{context} Can we say "{source}"?',
        )

@attr.s(auto_attribs=True)
class MRPCExampleTemplate(QNLIExampleTemplate):
    choices: list[str] = ["No", "Yes"]
    input_variables: list[str] = ['sentence2', 'sentence1', 'label']


cmsqa_prefix = "Answer the questions with one of the given choices:"
@attr.s(auto_attribs=True)
class CMSQAExampleTemplate(ClassificationExampleTemplate):
    choices: None = None
    prompt_format: str = 'Q-A'      # 'Q-A' or 'QC-A'
    input_variables: list[str] = ['question', 'choices', 'answerKey']

    def get_source(self, **kwargs):
        return kwargs[self.input_variables[0]].strip()

    def get_target(self, **kwargs):
        choices = self.get_choices(**kwargs)
        answer = choices[ord(kwargs[self.input_variables[2]])-ord('A')]
        return answer

    def get_choices(self, **kwargs):
        return [choice.strip() for choice in kwargs[self.input_variables[1]]['text']]

    @property
    def templates(self) -> dict[str, str]:
        if self.prompt_format == 'QC-A':
            return dict(
                train="Question: {source}\nChoices: {choices}\nAnswer: {target}",
                test="Question: {source}\nChoices: {choices}\nAnswer: "
            )
        else:
            return dict(
                train="Question: {source}\nAnswer: {target}",
                test="Question: {source}\nAnswer: "
            )

    def format(self, test=False, embedding=False, **kwargs):
        source = self.get_source(**kwargs)
        choices = self.get_choices(**kwargs)
        target = self.get_target(**kwargs)
        if embedding: return source
        template = self.templates['test'] if test else self.templates['train']
        return template.format(source=source, choices=', '.join(choices), target=target)

    def parse_output(self, lm_output: str, **kwargs) -> str:
        return lm_output.strip()

    def undo_format(self, string):
        raise NotImplementedError

if False:
    @attr.s(auto_attribs=True)
    class ContextualizeGenerationExampleTemplate(GenerationExampleTemplate):
        input_variables: list[str] = ['input', 'context', 'output']
        context_label: str = 'Context'
        version: str = 'with_context'
        embed_context: bool = False

        def get_context(self, **kwargs):
            return kwargs[self.input_variables[1]].strip()

        @property
        def template(self):
            assert self.version == 'with_context', 'ContextualizedClassificationExampleTemplate only works with prompt version="with_context"'
            return common_templates[self.version]

        def get_embedding_text(self, source, context):
            return source if not self.embed_context else f'{context}\n{source}'

        def format(self, test=False, embedding=False, **kwargs):
            source = self.get_source(**kwargs)
            context = self.get_context(**kwargs)
            target = self.get_target(**kwargs)
            if embedding:
                return self.get_embedding_text(source, context)
            template = self.templates['test'] if test else self.templates['train']
            return template.format(
                source=source, context=context, target=target,
                input_label=self.input_label,
                context_label=self.context_label,
                output_label=self.output_label
            )


    @attr.s(auto_attribs=True)
    class ContextualizedClassificationExampleTemplate(ClassificationExampleTemplate):
        input_variables: list[str] = ['text', 'context', 'label']
        context_label: str = 'Context'
        version: str = 'with_context'
        embed_context: bool = False

        def get_context(self, **kwargs):
            return kwargs[self.input_variables[1]].strip()

        @property
        def template(self):
            assert self.version == 'with_context', 'ContextualizedClassificationExampleTemplate only works with prompt version="with_context"'
            return common_templates[self.version]

        def get_embedding_text(self, source, context):
            return source if not self.embed_context else f'{context}\n{source}'

        def format(self, test=False, embedding=False, **kwargs):
            source = self.get_source(**kwargs)
            context = self.get_context(**kwargs)
            target = self.get_target(**kwargs)
            if embedding:
                return self.get_embedding_text(source, context)
            template = self.templates['test'] if test else self.templates['train']
            return template.format(
                source=source, context=context, target=target,
                input_label=self.input_label,
                context_label=self.context_label,
                output_label=self.output_label
            )

