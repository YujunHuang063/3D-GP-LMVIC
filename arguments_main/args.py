import sys
from argparse import ArgumentParser

from arguments_main import *

def get_dataset_args():
    parser = ArgumentParser(description="Dataset parameters")
    gaussian_model_params = GaussianModelParams(parser)
    gaussian_pipeline_params = GaussianPipelineParams(parser)
    dataset_params = DatasetParams(parser)
    training_params = TrainingParams(parser)
    model_params = ModelParams(parser)
    args_to_parse = filter_args(sys.argv[1:], parser)
    args = parser.parse_args(args_to_parse)
    return args

def get_main_args():
    parser = ArgumentParser(description="Main parameters")
    dataset_params = DatasetParams(parser)
    training_params = TrainingParams(parser)
    testing_params = TestingParams(parser)
    model_params = ModelParams(parser)
    general_params = GeneralParams(parser)
    args_to_parse = filter_args(sys.argv[1:], parser)
    args = parser.parse_args(args_to_parse)
    return args

def filter_args(args, parser):
    """Filter out args not defined in the parser."""
    parser_args = {}
    for action in parser._actions:
        option_strings = action.option_strings
        for option in option_strings:
            parser_args[option] = action

    filtered_args = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in parser_args:
            filtered_args.append(arg)
            action = parser_args[arg]
            nargs = action.nargs
            if nargs in [None, '?']:
                if i + 1 < len(args) and not args[i + 1].startswith('-'):
                    filtered_args.append(args[i + 1])
                    i += 1
            elif nargs == '*':
                j = i + 1
                while j < len(args) and not args[j].startswith('-'):
                    filtered_args.append(args[j])
                    j += 1
                i = j - 1
            elif nargs == '+':
                j = i + 1
                while j < len(args) and not args[j].startswith('-'):
                    filtered_args.append(args[j])
                    j += 1
                i = j - 1
            elif isinstance(nargs, int):
                for j in range(nargs):
                    if i + 1 + j < len(args) and not args[i + 1 + j].startswith('-'):
                        filtered_args.append(args[i + 1 + j])
                    else:
                        break
                i += nargs
        i += 1

    return filtered_args