import pandas as pd
import numpy as np
from collections import OrderedDict
from collections import namedtuple
from itertools import product
import time
from datetime import datetime
import json
import torch
from scipy.stats import spearmanr
import random
from torchvision.utils import make_grid
from uuid import uuid1

class RunBuilder():
    @staticmethod
    def grid_search(params):

        Run = namedtuple('Run', params.keys())

        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs
    
    @staticmethod
    def random_search(n_samples, params):

        Run = namedtuple('Run', params.keys())
        runs = []
        for s in range(n_samples):
            vs = []
            for k, v in params.items():
                if len(v) == 1:
                    v_tmp = v[0]

                if len(v) == 2:
                    if isinstance(v[0],int):
                        if k in ['channel_size', 'hidden_size']:
                            v_tmp = 2**int(np.random.randint(np.log2(np.amin(v)), np.log2(np.amax(v))+1,1)[0])
                        else:
                            v_tmp = int(np.random.randint(np.amin(v), np.amax(v)+1,1)[0])
                    elif isinstance(v[0],str):
                        v_tmp = random.choice(v)
                    elif isinstance(v[0], float):
                        v_tmp = np.round(np.random.uniform(np.amin(v), np.amax(v),1)[0],3)

                if len(v) > 2:
                    v_tmp = random.choice(v)

                vs.append(v_tmp)
            runs.append(Run(*vs))
        return runs


def get_hparam_runs(training_params, hparams):

    if training_params.hyperparameter_search == 'random':
            hparam_runs = RunBuilder.random_search(training_params.n_samples, vars(hparams))
            print('Random search:')
    if training_params.hyperparameter_search == 'grid':
            hparam_runs = RunBuilder.grid_search(vars(hparams))
            print('Grid search:')

    return hparam_runs

class Logger():

    def __init__(self):

        self.epoch_count = 0
        self.min_loss_epoch = 0
        self.epoch_collected_train_loss = []
        self.epoch_validation_loss = 0
        self.epoch_collected_validation_loss = []
        self.epoch_start_time = None
        
        self.run_params = None
        self.run_count = 0
        self.run_data = []
        self.run_start_time = None
        self.run_min_validation_loss = float('inf')

        self.train_loader = None
        self.validation_loader = None

        
    def begin_run(self, fname, hparam, path):

        self.run_start_time = time.time()
        self.now = datetime.now()
        date = self.now.strftime('%Y/%m/%d %H:%M:%S')
        self.ident = str(uuid1())
        self.fname = fname+f'_{self.ident}'
        self.path = path
        self.comment = f' {self.fname}'
        self.logdir = f'{path}/{date}' 
        self.run_params = hparam
        self.run_count += 1
        self.run_min_validation_loss = float('inf')
        
        self.hparam = hparam
 

        self.run_data = []
        self.epoch_collected_train_loss = []
        self.epoch_collected_validation_loss = []


    def end_run(self, paths_dict, training_options_dict):
        

        results = OrderedDict()
        results["id"] = self.ident
        results["model_name"] = training_options_dict['model_name']
        results["date"] = self.now.strftime('%Y/%m/%d %H:%M:%S')
        results["run"] = self.run_count
        results["n_epochs"] = self.epoch_count
        results['training loss'] = self.epoch_collected_train_loss
        results['validation loss'] = self.epoch_collected_validation_loss
        results['min validation loss'] = self.run_min_validation_loss
        results['run duration'] = time.time() - self.run_start_time
        results['paths'] = paths_dict
        results['training_options'] = training_options_dict
        results['hyperparameters'] = self.run_params._asdict()

        self.run_data.append(results)
        
        self.epoch_count = 0
        self.min_loss_epoch = 0
        del self.epoch_collected_train_loss 
        del self.epoch_collected_validation_loss


    def read_loader(self, train_loader, validation_loader):
        self.train_loader = train_loader
        self.validation_loader = validation_loader

    def begin_epoch(self):
        self.epoch_start_time = time.time()
        self.epoch_count += 1
        self.epoch_train_loss = 0
        self.epoch_validation_loss = 0

        
    def end_epoch(self):

        train_loss = self.epoch_train_loss / len(self.train_loader)
        self.epoch_collected_train_loss.append(train_loss)
        validation_loss = self.epoch_validation_loss / len(self.validation_loader)
        self.epoch_collected_validation_loss.append(validation_loss)

        if self.run_min_validation_loss > validation_loss:
            self.run_min_validation_loss = validation_loss

    def get_validation_loss(self):
        return self.epoch_validation_loss / len(self.validation_loader)


    def get_training_loss(self):
        return self.epoch_train_loss / len(self.train_loader)


    def get_run_count(self):
        return self.run_count


    def get_progress(self):
        return f'Epoch {self.epoch_count}/{self.run_params.n_epochs}'


    def get_uuid(self):
        return f'{self.ident}'


    def track_train_loss(self, loss):
        self.epoch_train_loss += loss.item()
        
        
    def track_validation_loss(self, loss):
        self.epoch_validation_loss += loss.item()


    def prediction_test(self, y, yhat):
        if y.shape[-1] > 1:
            for i in range(0, min(16, y.shape[0])):
                grid = make_grid(torch.cat([y[i:i+1], yhat[i:i+1]]), normalize=True) 

    def early_stopping(self, patience=20):
        stop_training = False
        validation_loss = self.epoch_validation_loss / len(self.validation_loader)
        if validation_loss < self.run_min_validation_loss:
            self.min_loss_epoch = self.epoch_count
        if self.epoch_count - self.min_loss_epoch > patience:
            stop_training = True 
        return stop_training
        

    def save_model(self, model, path):
        if self.epoch_count == 1:
            torch.save(model.state_dict(), f'{path}checkpoints/model_{self.ident}.pt')
        validation_loss = self.epoch_validation_loss / len(self.validation_loader)
        if validation_loss < self.run_min_validation_loss:
            torch.save(model.state_dict(), f'{path}checkpoints/model_{self.ident}.pt')


    def load_model(self, model, path):
        model.load_state_dict(torch.load(f'{path}checkpoints/model_{self.ident}.pt'))
        return model


def create_directory(path):
    """ Creates the directory for the download """
    import os
    if not os.path.isdir(path):
        try:
            os.makedirs(path+'/')
        except Exception as e:
            print(e)
        else:
            print('Path:', path, 'was created')


def set_seed():
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(999)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True


        