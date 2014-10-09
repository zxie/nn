import json
from run_utils import dump_config
from collections import defaultdict

SKIP_KEYS = ['descs', 'defaults']

class ParamStruct(object):
    '''
    Contains parameter data arrays saved during training
    '''

    def __init__(self, **entries):
        self.__dict__.update(entries)


class HyperparamStruct(object):
    '''
    Hyperparameter settings when running experiments
    '''

    def __init__(self, entries):
        self.descs = defaultdict(str)
        # Set self.defaults before calling super
        self.set_defaults()
        self.__dict__.update(entries)

    def add_to_argparser(self, parser):
        for k in self.__dict__:
            if k in SKIP_KEYS:
                continue
            # TODO Handle for parents too?
            argk = k
            if k in [d[0] for d in self.defaults]:
                argk = '--' + k
            parser.add_argument(argk, type=type(self.__dict__[k]), default=self.__dict__[k],
                    help=self.descs[k])

    def set_from_args(self, args):
        for (k, v) in args.__dict__.iteritems():
            if k in self.__dict__:
                self.__dict__[k] = v

    def add(self, key, default_value):
        self.__dict__[key] = default_value

    def add_desc(self, key, desc):
        assert key in self.__dict__
        self.descs[key] = desc

    def set_defaults(self):
        for default in self.defaults:
            self.__dict__[default[0]] = default[1]
            self.descs[default[0]] = default[2]


class ModelHyperparams(HyperparamStruct):
    # TODO Put common defaults here
    pass


def dump_to_json(hp_structs, out_file):
    '''
    Save hyperparameters for model, optimizer, etc. for later reference and loading
    Optimizer, model, etc. can have separate hyperparameter structs so not a method in class
    '''
    merged_dict = dict()
    for hp_struct in hp_structs:
        for k in hp_struct.__dict__:
            if k in SKIP_KEYS:
                continue
            # Check we don't have hyperparams with same name
            assert k not in merged_dict, 'Repeated hyperparam: %s' % k
            merged_dict[k] = hp_struct.__dict__[k]
    dump_config(merged_dict, out_file)


def load_from_json(hp_structs):
    pass
