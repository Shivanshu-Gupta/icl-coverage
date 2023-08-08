# https://github.com/timtadh/zhang-shasha
# https://github.com/mawilliams7/pyftk

import sys
import pickle
import math
import time
import nltk
from nltk.tree import Tree
from pathlib import Path

all_parse_trees = list()

class Memoize:
    def __init__(self, fn):
        self.fn = fn
        self.memo = {}

    def __call__(self, *args):
        if args not in self.memo:
            self.memo[args] = self.fn(*args)
        return self.memo[args]

def retrieve_parse_trees(parse_tree_filename):
    with open(parse_tree_filename, 'rb') as pkl_file:
        all_parse_trees = pickle.load(pkl_file)
    print("Parse trees retrieved.")
    return all_parse_trees

def extract_production_rules(tree, production_rules):
    left_side = tree.label()
    right_side = ""
    for subtree in tree:
        if type(subtree) == nltk.tree.Tree:
            right_side = right_side + " " + subtree.label()
            extract_production_rules(subtree, production_rules)
        else:
            right_side = right_side + " " + subtree
    production_rules.append((left_side + " ->" + right_side, tree))

def find_node_pairs(first_tree, second_tree):
    node_pairs = set()
    first_tree_production_rules = list()
    extract_production_rules(first_tree, first_tree_production_rules)
    first_tree_production_rules = sorted(first_tree_production_rules, key=lambda x : x[0])
    second_tree_production_rules = list()
    extract_production_rules(second_tree, second_tree_production_rules)
    second_tree_production_rules = sorted(second_tree_production_rules, key=lambda x : x[0])
    node_1 = first_tree_production_rules.pop(0)
    node_2 = second_tree_production_rules.pop(0)
    while node_1[0] != None and node_2[0] != None:
        if node_1[0] > node_2[0]:
            if len(second_tree_production_rules) > 0:
                node_2 = second_tree_production_rules.pop(0)
            else:
                node_2 = [None]
        elif node_1[0] < node_2[0]:
            if len(first_tree_production_rules) > 0:
                node_1 = first_tree_production_rules.pop(0)
            else:
                node_1 = [None]
        else:
            while node_1[0] == node_2[0]:
                second_tree_production_rules_index = 1
                while node_1[0] == node_2[0]:
                    node_pairs.add((str(node_1[1]), str(node_2[1])))
                    if second_tree_production_rules_index < len(second_tree_production_rules):
                        node_2 = second_tree_production_rules[second_tree_production_rules_index]
                        second_tree_production_rules_index += 1
                    else:
                        node_2 = [None]
                if len(first_tree_production_rules) > 0:
                    node_1 = first_tree_production_rules.pop(0)
                else:
                    node_1 = [None]
                if len(second_tree_production_rules) > 0:
                    node_2 = second_tree_production_rules[0]
                else:
                    node_2 = [None]
                if node_1[0] == None and node_2[0] == None:
                    break
    return node_pairs

def ftk(first_tree, second_tree):
    global all_parse_trees
    kernel_score = 0
    node_pairs = find_node_pairs(first_tree, second_tree)
    for node in node_pairs:
        if node[0] == node[1]:
            kernel_score += 1
    return kernel_score

def normalized_ftk(first_tree, second_tree):
    return ftk(first_tree, second_tree) / math.sqrt(ftk(first_tree, first_tree) * ftk(second_tree, second_tree))
