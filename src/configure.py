import pprint
import json

def params_from_file(key):
    import json
    with open('params.json') as json_file:
        params = json.load(json_file)
    params = Params(params[key])
    return params
        
class Params:
    def __init__(self, dictionary):
        for k, v in dictionary.items():
            setattr(self, k, v)

def print_congiguration(hparam_runs):

    paths = params_from_file('paths')
    training_params = params_from_file('training_params')
    hparams = params_from_file('hparams')

    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(vars(paths))
    pp.pprint(vars(hparams))
    pp.pprint(vars(training_params))
    pp.pprint(hparam_runs)
