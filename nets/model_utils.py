from rnn import RNN
from nplm import NPLM
from nclm import NCLM
from dnn import DNN

def get_model_class_and_params(model_type):
    model_type = model_type.lower()
    if model_type == 'dnn':
        model_class = DNN
    elif model_type == 'nplm':
        model_class = NPLM
    elif model_type == 'nclm':
        model_class = NCLM
    elif model_type == 'rnn':
        model_class = RNN
    else:
        raise Exception('Invalid model type: %s' % model_type)
    return model_class, model_class.init_hyperparams()
