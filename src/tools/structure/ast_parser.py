from typing import List, Any

def tokenize_lf(lf, add_sos=True):
    target = lf.replace(' [ ', '[').replace(' ]', ']').replace("(", " ( ").replace(")", " ) ").replace(",", " , ")
    if add_sos:
        target = f"<s> ( {target} )"
    tokens = target.split()
    return tokens

class ASTParser:
    _cached_ast_per_config = {}

    def __init__(self, config):
        self._config = config
        config_key = str(config)
        if config_key not in ASTParser._cached_ast_per_config:
            ASTParser._cached_ast_per_config[config_key] = {}

    def get_ast(self, input_norm: List[str]) -> List[Any]:
        config_key = str(self._config)
        cache_key = str(input_norm)
        if cache_key in ASTParser._cached_ast_per_config.get(config_key):
            return ASTParser._cached_ast_per_config[config_key][cache_key]

        ast = self._get_ast_rec(input_norm)
        ASTParser._cached_ast_per_config[config_key][cache_key] = ast

        return ast

    def _get_ast_rec(self, input_norm: List[str]) -> List[Any]:
        ast = []

        input_norm = [token for token in input_norm if token not in ['call', 'string', 'number']]

        elements = []
        current_element = []
        i = 0
        while i < len(input_norm):
            symbol = input_norm[i]
            if symbol == '(':
                list_content = []
                match_ctr = 1  # If 0, parenthesis has been matched.
                while match_ctr != 0:
                    i += 1
                    if i >= len(input_norm):
                        raise ValueError("Invalid input: Unmatched open parenthesis.")
                    symbol = input_norm[i]
                    if symbol == '(':
                        match_ctr += 1
                    elif symbol == ')':
                        match_ctr -= 1
                    # elif symbol == "," and match_ctr == 1:
                    #     elements.append(self.get_ast(list_content))
                    #     list_content = []
                    if match_ctr != 0:
                        list_content.append(symbol)
                current_element += self._get_ast_rec(list_content)
            elif symbol == ')':
                raise ValueError("Invalid input: Unmatched close parenthesis.")
            elif symbol == ',':
                elements.append(current_element)
                current_element = []
            else:
                # current_element.append(symbol)
                if current_element and isinstance(current_element[-1], str):
                    current_element[-1] += ' ' + symbol
                else:
                    current_element.append(symbol)

            i += 1
        elements.append(current_element)
        ast += elements

        return ast

def target_to_ast(target, verbose=False):
    from nltk import Tree
    tokens = tokenize_lf(target)
    ast_parser = ASTParser(config={})
    ast = ast_parser.get_ast(tokens)
    def post_process(_ast):
        if isinstance(_ast[0], list):
            # will ignore ast[1:]: eg. will ignore '[ NUMBER_VAL ]' in thingtalk
            return post_process(_ast[0]) if len(_ast[0]) > 0 else []
        elif isinstance(_ast, list):
            if len(_ast) == 1:
                return _ast[0]
            else:
                return Tree(_ast[0], [
                    post_process(c)
                    # if isinstance(c, list) and len(c) > 0 else c
                    if isinstance(c, list) else c
                    for c in _ast[1:] if c != []])
            # return [post_process(a) if isinstance(a, list) else a for a in _ast]
        else:
            raise ValueError(_ast)

    tree = post_process(ast)
    # tree = Tree.fromlist(post_process(ast))
    # tree = Tree.fromlist(ast[0])
    if verbose:
        tree.pretty_print(unicodelines=True)
    return tree