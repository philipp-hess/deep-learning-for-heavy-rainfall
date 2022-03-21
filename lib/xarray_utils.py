import xarray as xr
import numpy as np

def load(file_name:str,
        multi_files=False,
        rename=None,
        drop=None,
        extract=None,
        is_grib=False,
        chunk=None) -> xr.Dataset:
    
    """ Opens NetCDF for GRIB files and returns xarray.Dataset """
    
    if multi_files:
        ds = xr.open_mfdataset(file_name, chunks=chunk, combine="by_coords") 
        if is_grib:
            ds = xr.open_mfdataset(file_name, chunks=chunk, combine="by_coords", engine="cfgrib") 
    else:
        ds = xr.open_dataset(file_name, chunks=chunk) 
        if is_grib:
            ds = xr.open_dataset(file_name, chunks=chunk, engine="cfgrib") 

    if extract is not None:
        tmp = ds[extract]
        ds = tmp.to_dataset(name=extract)
        
    if drop is not None:
        ds = ds.drop(drop)

    if rename is not None:
        ds = ds.rename(rename)
        
    return ds


def reverse_latitudes(dataset: xr.Dataset) -> xr.Dataset:
    """ Reverses latitude coordinates """
    
    if "latitude" not in dataset.coords.keys():
        raise ValueError("Latitude is not a coordinate.") 
    dataset = dataset.reindex(latitude=list(reversed(dataset.latitude)))
    
    return dataset


def shift_longitudes(dataset: xr.Dataset) -> xr.Dataset:
    """
        Shifts longitude coordinates:
        from [0, 360] to [-180, 180]
        
    """
    if "longitude" not in dataset.coords.keys():
        raise ValueError("Longitude is not a coordinate.") 
    dataset = dataset.assign_coords(longitude=(((dataset.longitude + 180) % 360) - 180)).sortby('longitude')
    
    return dataset

def resample_to_daily_sums(dataset: xr.Dataset) -> xr.Dataset:
    """ Resamples to daily resolution.
        Can be used if sub-daily values are available at
        0 to 23h a daiy.
    """
    dataset = dataset.resample(time='D', closed='right').sum(dim='time')

    return dataset


def select_season(dataset: xr.Dataset, season='JJA') -> xr.Dataset:
    dataset = dataset.where(dataset["time"].dt.season==season, drop=True)
    return dataset
    

def crop_to_reference(dataset: xr.Dataset, ref_dataset: xr.Dataset) -> xr.Dataset:
    """ Crops horizontal coordinates to match reference dataset """
    
    if "longitude" not in dataset.coords.keys():
        raise ValueError("Longitude is not a coordinate of dataset.") 
    if "longitude" not in ref_dataset.coords.keys():
        raise ValueError("Longitude is not a coordinate of reference dataset.") 
    if "latitude" not in dataset.coords.keys():
        raise ValueError("Latitude is not a coordinate of dataset.") 
    if "latitude" not in ref_dataset.coords.keys():
        raise ValueError("Latitude is not a coordinate of reference dataset.") 
    
    dataset = dataset.where(dataset.latitude == ref_dataset.latitude, drop=True)\
                     .where(dataset.longitude == ref_dataset.longitude, drop=True)
    
    return dataset

def normalize_time(dataset: xr.Dataset) -> xr.Dataset:
    dataset['time'] = dataset.indexes['time'].normalize()
    return dataset

def sync(dataset_1: xr.Dataset, dataset_2: xr.Dataset):
    dataset_1 = dataset_1.where(dataset_1.time==dataset_2.time, drop=True)
    dataset_2 = dataset_2.where(dataset_2.time==dataset_1.time, drop=True)
    return dataset_1, dataset_2

def contains_nans(dataset: xr.DataArray)-> bool: 
    return np.isnan(np.sum(dataset.values))


def find_nan_time_frames(dataset: xr.DataArray)-> bool: 
    """" 
        Checks every time frame for NaNs and returns
        time stamp if NaN is found.
    """
    for i in range(len(dataset.time.values)):
        if np.isnan(np.sum(dataset.isel(time=i).values)):
            print(f'NaN at time index {i}')


def write_dataset(ds: xr.Dataset, file_name: str):
    import os
    from dask.diagnostics import ProgressBar
    print(f'writing to {file_name}')
    if os.path.isfile(f'{file_name}'):
        os.remove(f'{file_name}')
    delayed = ds.to_netcdf(f'{file_name}', compute=False)
    with ProgressBar():
        results = delayed.compute()

def regrid(ds: xr.Dataset, lats: list, lons: list,  method='bilinear', periodic=False) -> xr.Dataset:
    import xesmf as xe
    """
    Regrids xarray dataset.
    """

    grid_new = xr.Dataset({'lat': (['lat'], lats),
                           'lon': (['lon'], lons),
                          }
                         )
    ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})
    
    regridder = xe.Regridder(ds, grid_new, method, periodic=periodic)
    regridder.clean_weight_file()
    
    ds_new =  regridder(ds)
    ds_new = ds_new.rename({'lat': 'latitude', 'lon': 'longitude'})
    
    return ds_new 
