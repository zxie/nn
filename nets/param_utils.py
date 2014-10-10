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

    def set_from_dict(self, d):
        for (k, v) in d.iteritems():
            if k in self.__dict__:
                self.__dict__[k] = v

    def set_from_args(self, args):
        self.set_from_dict(args.__dict__)

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
