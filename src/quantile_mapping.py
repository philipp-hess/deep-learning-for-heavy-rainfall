import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from xclim import sdba

try:
    from evaluation_geographic import GeographicValidation 
except ModuleNotFoundError:
    from src.evaluation_geographic import GeographicValidation 

try:
    from evaluation_plots import plot_histogram
except ModuleNotFoundError:
    from src.evaluation_plots import plot_histogram

""" Script to run quantile mapping on the IFS test set data. """


def compute_ifs_quantile_mapping(path: str,
                                train_period: tuple,
                                test_period: tuple,
                                num_quantiles: int,
                                plot_hist=True) -> tuple:
    
    ds = xr.open_dataset(path, chunks={'time': 1})
    ifs = ds.ifs_total_precipitation.load()
    ifs_historical = ifs.sel(time=slice(str(train_period[0]), str(train_period[1])))
    ifs_test = ifs.sel(time=slice(str(test_period[0]), str(test_period[1])))

    trmm = ds.trmm_total_precipitation.load()
    trmm_reference = trmm.sel(time=slice(str(train_period[0]), str(train_period[1])))
    trmm_test = trmm.sel(time=slice(str(test_period[0]), str(test_period[1])))
    
    qm = sdba.EmpiricalQuantileMapping(nquantiles=num_quantiles)
    qm.train(trmm_reference, ifs_historical)
    ifs_quantile_mapped = qm.adjust(ifs_test)

    if plot_hist:
        plot_histogram(trmm_test.values, ifs_quantile_mapped.values, model_name='ifs', color='r')
        plot_histogram(trmm_test.values, trmm_test.values, model_name='trmm', color='k')
        plt.legend()
        plt.yscale('log')
        plt.show()
    
    eval = GeographicValidation(ds.latitude, ds.longitude)
    metrics_list = ['RMSE', 'Bias']
    mask_threshold = 0.025
    clean_threshold = 0.1
    ifs_qm, _, _ = eval.compute_metrics(metrics_list, ifs_quantile_mapped, ifs_quantile_mapped, trmm_test,
                 mask_threshold=mask_threshold, clean_threshold=clean_threshold)

    print(f'Mean RMSE: {ifs_qm["RMSE"].mean():3.3f}')
    print(f'Mean Bias: {ifs_qm["Bias"].mean():3.3f}')
    return ifs_quantile_mapped, trmm_test, ifs_test


def save(prediction: np.ndarray, target: np.ndarray, baseline: np.ndarray, fname: str):
    dir = '/path/to/models'
    tmp = np.stack([prediction, target, baseline])

    np.save(dir+'/'+fname, tmp)
    print('saved at ', dir+'/'+fname+'.npy')


def single_run():
    out_fname = 'qm.npy'
    num_quantiles = 750
    dataset_in_path = '/path/to/training_dataset.nc4'

    train_period = (2009, 2011)
    test_period = (2012, 2014)
    
    ifs_quantile_mapped, _, _ = compute_ifs_quantile_mapping(dataset_in_path, train_period,
                                                             test_period, num_quantiles)

    save(ifs_quantile_mapped, out_fname)


def parameter_scan():

    for year in range(1998, 2010):
        for num_quantiles in range(50,1050,50):
            print(year, num_quantiles)

            dataset_in_path = '/path/to/training_dataset.nc4'

            train_period = (year, 2011)
            test_period = (2012, 2014)

            _, _, _ = compute_ifs_quantile_mapping(dataset_in_path, train_period, test_period, num_quantiles,
                                                    plot_hist=False)

if __name__ == "__main__":
    #single_run()
    parameter_scan()