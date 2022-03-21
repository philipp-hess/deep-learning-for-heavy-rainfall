import numpy as np
import xarray as xr
import time

try:
    import evaluation_utils as eu
except ModuleNotFoundError:
    import src.evaluation_utils as eu

def load_data(fname):
    import pickle
    with open(f'{fname}.dat', "rb") as f:
        data = pickle.load(f)
    return data


def save_data(data, fname):
    import pickle
    with open(f'{fname}.dat', "wb") as f:
        pickle.dump(data, f)


def load_test_data(path_dict: dict) -> dict:
    results_dict = {}
    for model_name, path in path_dict.items():
        results_dict[model_name] = np.load(path)[0]
    results_dict['trmm'] = np.load(path)[1]
    results_dict['ifs'] = np.load(path)[2]
    return results_dict


def compute_thresholds(dataset_path: str,
                       percentiles: list,
                       min_precipitation_threshold_in_mm_per_3hours: float,
                       time_period=None) -> tuple:

    ds = xr.open_dataset(dataset_path, chunks={'time': 1})

    if time_period is not None:
        ds = ds.sel(time=slice(time_period[0], time_period[1]))

    data = ds.trmm_total_precipitation.values
    thresholds = {}
    masks = []

    for percentile in percentiles:

        print(f'computing {percentile}th threshold')
        threshold = eu.local_thresholds_from_percentiles(data, percentile,
                                                     data_min=min_precipitation_threshold_in_mm_per_3hours)
        thresholds[str(percentile)]=threshold

        masks.append(eu.get_threshold_mask(data, percentile,
                                data_min=min_precipitation_threshold_in_mm_per_3hours))
        
    return thresholds, masks
    

def transform_to_categorical_data(test_data: dict, thresholds: dict) -> dict:
    categorical_test_data = {}
    thresholds = [*thresholds.values()]
    for name, data in test_data.items():
        print(f'computing categorical for {name}')
        categorical_test_data[name] = eu.continuous_to_categorical_with_thresholds(data, thresholds)
    return categorical_test_data    


def compute_scores(categorical_test_data: dict, metrics: list, mask: np.ndarray) -> dict:
    all_scores = {}
    for name, data in categorical_test_data.items():
        if name != 'trmm':
            scores = {}
            for metric in metrics:
                start_time = time.time()
                scores[metric] = eu.categorical_evaluation(data, categorical_test_data['trmm'], metric, mask=mask[0])
                print('computing', name, metric, f'time: {(time.time() - start_time)/60: 2.1f} min')
        all_scores[name] = scores
    return all_scores


def relative_improvement(actual, base):
    return (np.array(actual)-np.array(base))/np.array(base) * 100