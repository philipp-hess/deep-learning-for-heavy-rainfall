import torch
from torch.utils.data import Dataset
import numpy as np
import xarray as xr
from torch.utils.data import DataLoader
import zarr
import os

def get_dataloader_training(paths, training_params, batch_size, device=None, uuid=None, caching_mode=None):
    """ 
        Args:
            paths (dict):
                I/O paths
            training_params (dict):
                conatains various training parameters
            batch_size (int)
            device (str):
                'cpu' or 'cuda'
            uuid (str):
                model identifier
            caching_mode (str):
                options: 'write' - writes a batch to disk,
                         'load'  - loads packed batches from disk,
                          None   - disables caching
        Returns:
            train_loader (DataLoader)
    """

    feature_dict = training_params['features']
    target_dict = {training_params['target']: None}
    target_str = training_params['target']
    num_workers = training_params['num_workers']

    
    if paths['input_format']  == 'netcdf':

        train_dataset = DaskDataset('training', feature_dict, target_str, paths,
                                     target_transform=training_params["target_transform"], 
                                     feature_transform=training_params["feature_transform"],
                                     lazy=training_params['lazy'])

        train_loader = DataLoader(train_dataset,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     num_workers=num_workers,
                                     drop_last=True,
                                     pin_memory=torch.cuda.is_available())

    if paths['input_format']  == 'zarr':

        if caching_mode == 'write':

            train_dataset = CachingZarrDataset('training', feature_dict, target_str, paths,
                                        feature_transform=training_params["feature_transform"],
                                        write_cache=True,
                                        load_cache=False,
                                        batch_size=batch_size,
                                        device=device,
                                        uuid=uuid)

            train_loader =  DataLoader(train_dataset,
                                       batch_size=batch_size,
                                       drop_last=True,
                                       num_workers=0, 
                                       shuffle=True)

        if caching_mode == 'load':

            train_dataset = CachingZarrDataset('training', feature_dict, target_str, paths,
                                        feature_transform=training_params["feature_transform"],
                                        write_cache=False,
                                        load_cache=True,
                                        batch_size=batch_size,
                                        device=device,
                                        uuid=uuid)

            train_loader =  DataLoader(train_dataset,
                                       batch_size=None,
                                       drop_last=False,
                                       num_workers=0,
                                       shuffle=False)

        if caching_mode is None:

            train_dataset = ZarrDataset('training', feature_dict, target_str, paths,
                                        target_transform=training_params["target_transform"],
                                        feature_transform=training_params["feature_transform"])

            train_loader = DataLoader(train_dataset,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         num_workers=num_workers,
                                         drop_last=True,
                                         pin_memory=torch.cuda.is_available())

    return train_loader


def get_dataloader_validation(paths, training_params, batch_size, device=None, uuid=None, caching_mode=None):
    """ 
        Args:
            paths (dict):
                I/O paths
            training_params (dict):
                conatains various training parameters
            batch_size (int)
            device (str):
                'cpu' or 'cuda'
            uuid (str):
                model identifier
            caching_mode (str):
                options: 'write' - writes a batch to disk,
                         'load'  - loads packed batches from disk,
                          None   - disables caching
        Returns:
            validation_loader (DataLoader)
    """

    feature_dict = training_params['features']
    target_dict = {training_params['target']: None}
    target_str = training_params['target']
    num_workers = training_params['num_workers']

    
    if paths['input_format']  == 'netcdf':

        validation_dataset = DaskDataset('validation',  feature_dict, target_str, paths,
                                         target_transform=training_params["target_transform"],
                                         feature_transform=training_params["feature_transform"],
                                         lazy=training_params['lazy'])

        validation_loader = DataLoader(validation_dataset,
                                   batch_size=batch_size,
                                   shuffle=False,
                                   num_workers=num_workers,
                                   drop_last=False,
                                   pin_memory=torch.cuda.is_available())

    if paths['input_format']  == 'zarr':

        if caching_mode == 'write':


            validation_dataset = CachingZarrDataset('validation',  feature_dict, target_str, paths,
                                            feature_transform=training_params["feature_transform"],
                                            write_cache=True,
                                            load_cache=False,
                                            batch_size=batch_size,
                                            device=device,
                                            uuid=uuid)

            validation_loader =  DataLoader(validation_dataset,
                                            batch_size=batch_size,
                                            drop_last=True,
                                            num_workers=0, 
                                            shuffle=True)


        if caching_mode == 'load':

            validation_dataset = CachingZarrDataset('validation',  feature_dict, target_str, paths,
                                            feature_transform=training_params["feature_transform"],
                                            write_cache=False,
                                            load_cache=True,
                                            batch_size=batch_size,
                                            device=device,
                                            uuid=uuid)

            validation_loader =  DataLoader(validation_dataset,
                                           batch_size=None,
                                           drop_last=False,
                                           num_workers=0,
                                           shuffle=False)

        if caching_mode is None:

            validation_dataset = ZarrDataset('validation',  feature_dict, target_str, paths,
                                            target_transform=training_params["target_transform"], 
                                            feature_transform=training_params["feature_transform"])

            validation_loader = DataLoader(validation_dataset,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       num_workers=num_workers,
                                       drop_last=False,
                                       pin_memory=torch.cuda.is_available())

    return validation_loader


def get_dataloader_test(paths, training_params, batch_size, device=None, uuid=None, caching_mode=None):
    """ 
        Args:
            paths (dict):
                I/O paths
            training_params (dict):
                conatains various training parameters
            batch_size (int)
            device (str):
                'cpu' or 'cuda'
            uuid (str):
                model identifier
            caching_mode (str):
                options: 'write' - writes a batch to disk,
                         'load'  - loads packed batches from disk,
                          None   - disables caching
        Returns:
            train_loader (DataLoader)
            validation_loader (DataLoader)
    """

    feature_dict = training_params['features']
    target_str = training_params['target']

    
    if paths['input_format']  == 'netcdf':

        test_dataset = DaskDataset('test', feature_dict, target_str, paths,
                                     target_transform=training_params["target_transform"], 
                                     feature_transform=training_params["feature_transform"],
                                     lazy=training_params['lazy'])

        test_loader = DataLoader(test_dataset,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     num_workers=0,
                                     drop_last=False,
                                     pin_memory=torch.cuda.is_available())

    if paths['input_format']  == 'zarr':

        test_dataset = ZarrDataset('test', feature_dict, target_str, paths,
                                    target_transform=training_params["target_transform"],
                                    feature_transform=training_params["feature_transform"])

        test_loader = DataLoader(test_dataset,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     num_workers=0,
                                     drop_last=False,
                                     pin_memory=torch.cuda.is_available())

    return test_loader


def count_features(feature_dict):
    n_features = 0
    for var, levels in feature_dict.items():
            if levels:
                n_features += len(levels)
            else :
                n_features += 1
    return n_features

class DatasetStandardization():

    def __init__(self, flag, feature_dict, target_str, params, paths):
        """
        Splits the netcdf dataset into training, validation and test set
        and standardizes the features to zero mean and unit standard deviation.


        Args:
            flag (str): 'training', 'validation' or 'test'
            feature_dict (dict):
                contaings feautures as keys and the pressure levels as values
            target_str (str):
                name of target
            params (dict):
                contains training parameters
            paths (dict):
                contains all directory paths
        """

        print(f'\n Preprocess {flag} data:')
        self.paths = paths
        self.flag = flag
        dataset = self.get_netcdf_data(flag, params, paths)
        self.features = self.get_features(dataset, feature_dict, params, paths)
        self.target_str = target_str
        self.target = self.get_target(dataset, target_str)


    def get_netcdf_data(self, flag, params, paths):
        """ Loads the training data stored in NETCDF format

            Args:
                flag (str): 'training', 'validation' or 'test'
                params (dict): dictionary with keys <flag>_start, <flag>_end
                paths (dict): dictionary with keys 'dataset_path', 'dataset_<flag>'
            Returns:
                dataset (xr.Dataset): dataset containing freatures and target
        """
        print('loading netcdf file..')
        with xr.open_dataset(f'{paths["dataset_path"]}/{paths["dataset_training"]}',
                             chunks={'time':1}) as dataset:
            if flag == 'training':
                dataset = dataset.sel(time=slice(str(params['training_start']),
                                                 str(params['training_end'])))
            if flag == 'validation':
                dataset = dataset.sel(time=slice(str(params['validation_start']),
                                                 str(params['validation_end'])))
            if flag == 'test':
                dataset = dataset.sel(time=slice(str(params['test_start']),
                                                 str(params['test_end'])))
        return dataset 


    def get_target(self, dataset, target_str):
        """ Extracts the training target netcdf dataset.

            Args:
                dataset (xr.Dataset): dataset containing freatures and target
                target_str (str): dictionary containing the target as key as 
                                    pressure level or None as value.
            Returns:
                target (toch.Tensor): training target of shape [samples, 1, lats, lons]
        """

        target = dataset[target_str].transpose('time', 'latitude', 'longitude')

        return target


    def get_features(self, dataset, feature_dict, params, paths):
        """ Extracts the training features and standardizes.

            Args:
                dataset (xr.Dataset): dataset containing freatures and target.
                features_dict (dict): dictionary containing the features as key as 
                                      pressure levels as value.
                params (dict): dictionary with keys training_start, training_end
                paths (dict): dictionary with keys 'dataset_training', 'dataset_training'
            Returns:
                features (toch.Tensor): training features of shape [samples, features, lats, lons]
        """

        print('extracting features..')

        dataset_feature = dataset[list(feature_dict.keys())]

        dataset_training = xr.open_dataset(f'{paths["dataset_path"]}/{paths["dataset_training"]}', chunks={'time':92})
        dataset_training = dataset_training.sel(time=slice(str(params['training_start']),
                                                           str(params['training_end'])))

        mean = dataset_training[list(feature_dict.keys())].mean(('time', 'latitude', 'longitude')).compute()
        std = dataset_training[list(feature_dict.keys())].std(('time', 'latitude', 'longitude')).compute()

        dataset_feature = (dataset_feature - mean)/std

        dataset_training.close()

        print(f'features standardized.')
        data = []
        dummy_level = xr.DataArray([1], coords={'level': [1]}, dims=['level'])
        n_features = 0
        for var, levels in feature_dict.items():
            try:
                data.append(dataset_feature[var].sel(level=levels))
                n_features += len(levels)
            except ValueError:
                n_features += 1
                data.append(dataset_feature[var].expand_dims({'level': dummy_level}, 1))

        features = xr.merge(data).transpose('time', 'level', 'latitude', 'longitude')

        del data, dataset_feature

        return features

    def write_to_file(self, format='netcdf'):
        from dask.diagnostics import ProgressBar
        
        combined_data = xr.merge([self.features, self.target])

        if format == 'netcdf':
            out_filename = f'{self.paths["dataset_path"]}/standardized/{self.paths["dataset_training"][:-4]}_target_{self.target_str}_{self.flag.upper()}.nc4'

            delayed = combined_data.to_netcdf(out_filename, compute=False)
            with ProgressBar():
                results = delayed.compute()

            print(f'created: {out_filename}')

        if format == 'zarr':
            out_filename = f'{self.paths["dataset_path"]}/standardized/{self.paths["dataset_training"][:-4]}_target_{self.target_str}_{self.flag.upper()}.zarr'

            for var in combined_data:
                combined_data[var].encoding.pop('scale_factor', None)
                combined_data[var].encoding.pop('add_offset', None)
                combined_data[var].encoding['dtype'] = np.dtype('float32')

            combined_data.to_zarr(out_filename, mode='w')
            print(f'created: {out_filename}')


class DaskDataset(torch.utils.data.Dataset):
    
    def __init__(self, flag, feature_dict, target_name, paths,
                 target_transform=None, feature_transform=None, lazy=True):
        """
        Dataset using memory maps to reduce RAM consumtion.
        
        Args:
            file_path (str):
                path and file name of training data
            features (dict):
            target_name (str):
        Note:
            Coordinate names in netcdf file have to be:
            'time', 'level', 'latitude', 'longitude'
        """
        
        self.target_name = target_name
        self.target_transform = target_transform
        self.feature_transform = feature_transform

        if self.target_transform  == 'log':
            print("using target log-transform")

        filename = f'{paths["dataset_path"]}/standardized/{paths["dataset_training"][:-4]}_target_{target_name}_{flag.upper()}.nc4'
        if lazy:
            self.ds = xr.open_dataset(filename, chunks={'time': 1})
        else:
            self.ds = xr.open_dataset(filename, cache=True)
        
        data = []
        for var, levels in feature_dict.items():
            if levels is not None:
                data.append(self.ds[var].sel(level=levels))
            else:
                data.append(self.ds[var].sel(level=1))
                
        self.features = xr.concat(data, 'level').transpose('time', 'level', 'latitude', 'longitude')
        
        self.target = self.ds[target_name].transpose('time', 'latitude', 'longitude')
        
    def __getitem__(self, index):
        
        features_sample = torch.from_numpy(self.features.isel(time=index).values).float()
        target_sample = torch.from_numpy(self.target.isel(time=index).values).float()

        if self.target_transform == 'log':
            epsilon = 0.001
            target_sample = torch.log(target_sample + epsilon) - np.log(epsilon)

        if self.target_transform == 'log':
            epsilon = 0.001
            target_sample = torch.log(target_sample + epsilon) - np.log(epsilon)

        if self.feature_transform == 'linear':
            nlats = target_sample.shape[-1]
            nlons = target_sample.shape[-2]
            n_features = features_sample.shape[-3]
            target_sample = target_sample.reshape(nlats*nlons, 1)
            features_sample = features_sample.reshape(nlats*nlons, n_features)
            target_sample = target_sample
        else:
            target_sample = target_sample.unsqueeze(0)
        
        return features_sample, target_sample

    def __len__(self):
        return len(self.features.time) 


class ZarrDataset(torch.utils.data.Dataset):
    
    def __init__(self, flag, feature_dict, target_name, paths,
                target_transform=None, feature_transform=None):
        """
        Dataset using memory maps for larger than memory dataset.
        
        Args:
            file_path (str):
                path and file name of training data
            features (dict):
            target_name (str):
        Note:
            Coordinate names in netcdf file have to be:
            'time', 'level', 'latitude', 'longitude'
        """
        
        filename = f'{paths["dataset_path"]}/standardized/{paths["dataset_training"][:-4]}_{flag.upper()}.zarr'
        store = zarr.DirectoryStore(filename)
        self.ds = zarr.open(store, mode='r')
        self.n_lat = len(self.ds['latitude'])
        self.n_lon = len(self.ds['longitude'])
        self.feature_transform = feature_transform
        self.target_name = target_name

        data, self.level_indices = [], []
        for var, levels in feature_dict.items():

            if levels is None: 
                level_index=0
            else:
                all_level = self.ds['level'][:]
                level_index = list(np.where(np.isin(all_level, levels))[0])

            data.append(self.ds[var])
            self.level_indices.append(level_index)
        
        self.features = data
        self.target = self.ds[target_name]
        
    def __getitem__(self, index):
        
        data  = []
        for var, l in zip(self.features, self.level_indices):
            if l == 0:
                tmp = torch.from_numpy(var.oindex[index, l].reshape(1, self.n_lat,  self.n_lon))
                data.append(tmp)
            else:
                tmp = torch.from_numpy(var.oindex[index, l])
                data.append(tmp)

        features_sample = torch.cat(data).float()

        features_sample = features_sample.view(-1, self.n_lat, self.n_lon)

        target_sample = torch.Tensor(self.target[index]).float()

        if self.feature_transform == 'linear':
            nlats = target_sample.shape[-1]
            nlons = target_sample.shape[-2]
            n_features = features_sample.shape[-3]
            target_sample = target_sample.reshape(nlats*nlons, 1)
            features_sample = features_sample.reshape(nlats*nlons, n_features)

        else:
            target_sample = target_sample.unsqueeze(0)
        

        return features_sample, target_sample

    def __len__(self):
        return self.features[0].shape[0] 


class CachingZarrDataset(torch.utils.data.Dataset):
    
    def __init__(self, flag, feature_dict, target_name, paths,
                feature_transform=None,
                write_cache=False,
                load_cache=False,
                batch_size=1,
                device=None,
                uuid=None
                ):
        """
        Dataset using Zarr datastore for larger than memory dataset.
        
        Args:
            file_path (str):
                path and file name of training data
            features (dict):
            target_name (str):
        Note:
            Coordinate names in netcdf file have to be:
            'time', 'level', 'latitude', 'longitude'
        """

        filename = f'{paths["dataset_path"]}/standardized/{paths["dataset_training"][:-4]}_{flag.upper()}.zarr'
        store = zarr.DirectoryStore(filename)
        self.ds = zarr.open(store, mode='r')
        self.n_lat = len(self.ds['latitude'])
        n_time = len(self.ds['time'])
        #n_time = 20
        self.n_lon = len(self.ds['longitude'])
        self.feature_transform = feature_transform
        self.target_name = target_name

        data, self.level_indices = [], []
        for var, levels in feature_dict.items():
            if levels is None: 
                level_index=0
            else:
                all_level = self.ds['level'][:]
                level_index = list(np.where(np.isin(all_level, levels))[0])
            data.append(self.ds[var])
            self.level_indices.append(level_index)
        
        self.features = data
        self.target = self.ds[target_name]

        # --------------- caching parameters --------------
        self.device = device
        if batch_size is not None:
            self.batch_index = np.arange(1, n_time/batch_size+1)
        self.cache_dir = f'{paths["cache_path"]}/{uuid}/{flag}'
        self.write_cache = write_cache
        self.load_cache = load_cache
        if not self.load_cache:
            self.n_samples = n_time
        else:
            self.n_samples = int(np.floor(n_time/batch_size))
        self.count_samples = 0
        self.count_batches = 0
        self.batch_size = batch_size
        self.feature_cache = []
        self.target_cache = []
        # -------------------------------------------------
        
    def __getitem__(self, index):
        
        if self.load_cache:
            features_sample, target_sample = self.read_cache()
        else:
            data  = []
            for var, l in zip(self.features, self.level_indices):
                if l == 0:
                    tmp = torch.from_numpy(var.oindex[index, l].reshape(1, self.n_lat,  self.n_lon))
                    data.append(tmp)
                else:
                    tmp = torch.from_numpy(var.oindex[index, l])
                    data.append(tmp)

            features_sample = torch.cat(data).view(-1, self.n_lat, self.n_lon).float()
            target_sample = torch.Tensor(self.target[index]).float()

            if self.feature_transform == 'linear':
                n_features = features_sample.shape[-3]
                target_sample = target_sample.reshape(self.n_lat*self.n_lon, 1)
                features_sample = features_sample.reshape(self.n_lat*self.n_lon, n_features)
            else:
                target_sample = target_sample.unsqueeze(0)
        
         
        if self.write_cache:
            self.feature_cache.append(torch.Tensor(features_sample))
            self.target_cache.append(torch.Tensor(target_sample))
            self.save_cache()
        self.count()


        return features_sample, target_sample

    def __len__(self):
        return self.n_samples
    

    def count(self): 
        self.count_samples += 1
        if self.count_samples == self.batch_size and not self.load_cache : 
            self.count_samples = 0
            self.feature_cache = []
            self.target_cache = []
            self.count_batches += 1
        if self.load_cache:
            self.count_batches += 1
   

    def save_cache(self):
        if len(self.feature_cache) == self.batch_size: 
            # create cache dir if non-existent
            if self.count_batches == 0:
                if not os.path.isdir(self.cache_dir):
                    os.makedirs(self.cache_dir)
            feature = torch.stack(self.feature_cache)
            target = torch.stack(self.target_cache)
            torch.save(feature, f'{self.cache_dir}/feature_{self.count_batches}.pt')
            torch.save(target, f'{self.cache_dir}/target_{self.count_batches}.pt')
           
        
    def read_cache(self):
        feature = torch.load(f'{self.cache_dir}/feature_{self.count_batches}.pt', map_location=torch.device(self.device))
        target = torch.load(f'{self.cache_dir}/target_{self.count_batches}.pt', map_location=torch.device(self.device))
        return feature, target 

def clean_cache(path):
    import shutil
    import os
    if os.path.isdir(path):
        print('test', os.path.isdir(path))
        shutil.rmtree(path)
    