
import torch 
import torch.nn as nn

import src.run_utils as ru 
from src.run_utils import create_directory, set_seed
from src.models import load_model
from src.loss import LossFunctions
from src.training import training
from src.configure import params_from_file, print_congiguration


def main():
    """ Main method to execute the NN training. """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Running on', device)

    paths = params_from_file('paths')
    training_params = params_from_file('training_params')
    hparams = params_from_file('hparams')

    ru.create_directory(paths.tensorboard_path)
    hparam_runs = ru.get_hparam_runs(training_params, hparams)
    print_congiguration(hparam_runs)

    for counter, hparam in enumerate(hparam_runs):
        print(f"Run {counter+1}/{len(hparam_runs)} ")
        
        model = load_model(training_params, hparam, device=device)

        if torch.cuda.device_count() > 1:
                print("Using", torch.cuda.device_count(), "GPUs")
                model = nn.DataParallel(model)

        losses = LossFunctions(hparam, device=device)
        cost = getattr(losses, training_params.loss_function)
        
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=10**(float(-hparam.lr)),
                                     weight_decay=hparam.weight_decay)

        training(training_params, model, cost, optimizer, device, hparam, paths)

    print('Training finished.')    


if __name__ == "__main__":
    set_seed()
    main()
