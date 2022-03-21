import sys
sys.path.append("..")
import lib.xarray_utils as ut
import xarray as xr
from dataclasses import dataclass
from src.configure import params_from_file


def main():

    paths = params_from_file('paths')
    data = load_data(paths)
    data = process_trmm_data(data)
    data = crop(data)
    data = synchronize(data)
    dataset = merge(data)
    ut.write_dataset(dataset, paths.out_dataset)

def load_data(paths):
    ifs_vertical_velocity = ut.load(paths.ifs_vertical_velocity, multi_files=True, drop=None, chunk={'time': 1},
                                    rename={'w': 'vertical_velocity'})

    ifs_precipitation = ut.load(paths.ifs_precipitation, multi_files=True, drop=None, chunk={'time': 1},
                                rename={'tp': 'ifs_precipitation'})

    rename = {'precipitation': 'trmm_precipitation',
              'lat': 'latitude',
              'lon': 'longitude',
             }
    trmm_precipitation = ut.load(paths.trmm_precipitation, multi_files=False, extract='precipitation',
                                 chunk={'time': 1}, rename=rename)

    data = Data(ifs_vertical_velocity=ifs_vertical_velocity,
                ifs_precipitation=ifs_precipitation,
                trmm_precipitation=trmm_precipitation
                )

    return data

@dataclass
class Data:
    ifs_vertical_velocity: None
    ifs_precipitation: None
    trmm_precipitation: None

def crop(data):
    data.ifs_precipitation = ut.crop_to_reference(data.ifs_precipitation,
                                                  data.trmm_precipitation)

    data.ifs_vertical_velocity = ut.crop_to_reference(data.ifs_vertical_velocity,
                                                      data.trmm_precipitation)
    return data

def synchronize(data):
    data.ifs_precipitation, data.trmm_precipitation  = ut.sync(data.ifs_precipitation,
                                         data.trmm_precipitation)

    data.ifs_vertical_velocity , _  = ut.sync(data.ifs_vertical_velocity,
                                              data.trmm_precipitation)
    return data

def merge(data):
    data_list = [
                data.ifs_vertical_velocity,
                data.ifs_precipitation,
                data.trmm_precipitation,
            ]
    dataset = xr.merge(data_list)
    return dataset

def process_trmm_data(data):
    trmm = data.trmm_precipitation
    trmm = ut.select_season(trmm, season='JJA')
    trmm = trmm.transpose('time', 'latitude', 'longitude')
    data.trmm_precpipitation = trmm
    data.trmm_precipitation = data.trmm_precipitation.dropna('time', how='any')
    return data

if __name__ == '__main__':
    main()


