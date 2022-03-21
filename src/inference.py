import os

import numpy as np
import torch
import torch.nn as nn
import xarray as xr
from IPython.display import clear_output
from lib.spark import ModelData
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataclasses import dataclass

import src.evaluation_plots as eplt
from src.dataset import get_dataloader_test, get_dataloader_validation
from src.models import load_model
from lib.xarray_utils import reverse_latitudes, shift_longitudes, regrid



@dataclass
class InferenceConfig:

    model_data: ModelData
    model_id: str
    frequency: str 

    model_type: str = 'unet'
    dataset: str = 'test'
    batch_size: int = 1
    device: str = 'cpu'
    verbose: bool = False 
    data_parallel: bool = False
    num_workers: int = 0
    input_format: str = 'netcdf'
    data_parallel: bool = False


def model_inference(config: InferenceConfig):
    """
    Performes ANN inference on test set by loading the model saved during training.

    Args:
        config:
            Contains all congiguration parameter

    Returns:
        yhat (np.ndarray): shape (n_examples, n_lat, n_lon)
            model prediction
        y (np.ndarray): shape (n_examples, n_lat, n_lon)
            test target
        x (np.ndarray): shape (n_examples, n_features, n_lat, n_lon)
            input features
        model (nn.Module): 
            pytroch ANN model
     """
    
    training_options = config.model_data.get_training_options(config.model_id)
    paths =  config.model_data.get_training_paths(config.model_id)
    feature_dict =  training_options['features']
    target_dict =  {training_options['target']: None}

    df = config.model_data.get_model_hyperparams(config.model_id).to_dict('records')[0]
    hparam = MakeClass(df)

    if config.dataset == 'test':
        data_loader = get_dataloader_test(paths, training_options, hparam.batch_size, caching_mode=None)

    if config.dataset == 'validation':
        data_loader = get_dataloader_validation(paths, training_options, hparam.batch_size, caching_mode=None)

    model = load_model(MakeClass(training_options), hparam)

    if config.data_parallel:
        model = nn.DataParallel(model)
    fname = get_model_checkpoint(config.model_id)

    print(fname)
    checkpoint = torch.load(fname,  map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    model = model.to(config.device)
    
    print(f'Beginning inference...')

    model.eval() 
    x_list, y_list, yhat_list = [], [], []
    with torch.no_grad():
        for batch, (x, y) in enumerate(data_loader):
            x, y = x.to(config.device), y.to(config.device)

            if config.verbose:
                print(f'batch {batch+1}/{len(data_loader)}')
                clear_output(wait=True)

            yhat = model(x)

            y_list.append(y.cpu())
            yhat_list.append(yhat.cpu())
        
    yhat = torch.cat(yhat_list)
    y = torch.cat(y_list)

    del y_list, yhat_list

    y = y.squeeze(1).numpy()
    yhat = yhat.squeeze(1).numpy()

    return yhat, y, model


def get_baseline(config: InferenceConfig):
    """
    Get physical precipitation baseline.

    Args:
        config:
            Contains all congiguration parameter

    Returns:
        baseline_precipitation (np.ndarray): shape (n_examples, n_lat, n_lon)
            baseline model output
        lats (xr.DataArray): shape (n_lat)
            latitude coordinates for basemap plotting
        lons (xr.DataArray): shape (n_lon)
            longitude coordinates for basemap plotting
    """

    dataset_paths =  config.model_data.get_training_paths(config.model_id)
    start = config.model_data.get_training_options(config.model_id)[f'{config.dataset}_start']
    end = config.model_data.get_training_options(config.model_id)[f'{config.dataset}_end']

    ds = xr.open_dataset(dataset_paths['dataset_path']+'/'+dataset_paths['dataset_training'])
    lats = ds.latitude
    lons = ds.longitude

    baseline_precipitation = ds['ifs_total_precipitation']
    baseline_precipitation = baseline_precipitation.sel(time=slice(str(start), str(end)))
    
    return baseline_precipitation.values, lats, lons


def evaluate_models(config: InferenceConfig):
    """
    Valuates NN model based on
        - spatial averaged mean
        - RMSE of spatial average
        - frequency histogram
        Args:
            config:
                Contains all congiguration parameters
        Returns:
            lats (xarray.DataArray): shape [n_lats]
            lons (xarray.DataArray): shape [n_lons]
            prediction (numpy.ndarray): shape [n_time, n_lat, n_lon, 1]
            baseline (numpy.ndarray): shape [n_time, n_lat, n_lon, 1]
            target (numpy.ndarray): shape [n_time, n_lat, n_lon, 1]
            features(numpy.ndarray): shape [n_time, n_lat, n_lon, n_features]
            model (torch.nn.Model):
                trained model
    

    """
    from IPython.display import display 

    model_id = config.model_id
    hparams = config.model_data.get_model_hyperparams(model_id)        
    options = config.model_data.get_training_options(model_id)
    display(hparams)

    baseline, lats, lons = get_baseline(config)
    if config.frequency == '3hourly':
        baseline = baseline*1000
    
    prediction, target, model = model_inference(config)

    if config.model_type == 'linear':
        nlats = len(lats)
        nlons = len(lons)
        prediction = prediction.reshape(-1, nlats, nlons)
        target = target.reshape(-1, nlats, nlons)

    print(prediction.shape, target.shape, baseline.shape)
        
    eplt.plot_accuracy(prediction[:300], baseline[:300], target[:300],
                     baseline_name='ERA5-ensemble-mean',
                     title='Rainfall prediction error, target = TRMM',
                     metric='RMSE')
        
    eplt.plot_precipitation_mean(prediction[:300], baseline[:300], target[:300], '')

    ifs = eplt.PlotData(baseline, 'tab:blue', 'IFS')
    model = eplt.PlotData(prediction, 'tab:red', 'Model')    
    trmm = eplt.PlotData(target, 'k', 'TRMM')    
    data = [ifs, model, trmm]

    eplt.plot_precipitation_frequencies(target, data)
        
    return lats, lons, prediction, baseline, target, model

class MakeClass:
    def __init__(self, dictionary):
        for k, v in dictionary.items():
            setattr(self, k, v)


def save_prediction(name, prediction, target, baseline):
    dir='/path/to/models'

    tmp = np.stack([prediction, target, baseline])
    np.save(dir+'/'+name, tmp)
    print('saved at ', dir+'/'+name+'.npy')


def load_prediction(name):
    dir='/path/to/checkpoints'
    
    tmp = np.load(dir+'/'+name+'.npy')
    prediction, target, baseline = tmp[0], tmp[1], tmp[2]
    
    return prediction, target, baseline


def parse_file_name(fname, regex):
    import re
    pattern = re.compile(regex)
    file = pattern.search(fname)
    return file


def get_model_checkpoint(model_id):
    import fnmatch

    regex = model_id
    path = '/path/to/checkpoints'
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, '*.pt*'):
                f = os.path.join(root, name)
                if parse_file_name(f, f'{regex}') is not None:
                    file = f
    return file


class Mask():
    def __init__(self, data):
        """ 
        The land-sea mask can be downloaded at:
        https://confluence.ecmwf.int/download/attachments/140385202/lsm_1279l4_0.1x0.1.grb_v4_unpack.nc?version=1&modificationDate=1591979822208&api=v2 
        """ 
        self.mask_path = f'/path/to/lsm_1279l4_0.1x0.1.grb_v4_unpack.nc'
        self.dataset_path = f'/path/to/training_dataset.nc4'
        self.data = data

    def create_numpy_mask(self, tropics_min_latitude=None, tropics_max_latitude=None):
        
        mask = xr.open_dataset(self.mask_path)
        mask = reverse_latitudes(mask)
        mask = shift_longitudes(mask)
        
        ds = xr.open_dataset(self.dataset_path)
        mask = mask.lsm.sel(latitude=slice(ds.latitude.min(), ds.latitude.max()))\
                       .sel(longitude=slice(ds.longitude.min(), ds.longitude.max()))
        
        mask = regrid(mask, ds.latitude, ds.longitude)

        if tropics_max_latitude is not None: 
            mask = mask.where(mask.latitude < tropics_max_latitude,0)
            mask = mask.where(mask.latitude > tropics_min_latitude,0)
        
        return mask
    
    def apply(self):
        mask = self.create_numpy_mask()
        # mask is 0 on sea area and non-zero at land.
        data = np.where(mask>0, self.data, 0)
        return data