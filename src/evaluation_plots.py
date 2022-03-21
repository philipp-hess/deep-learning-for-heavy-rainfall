import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

def plot_score_over_percentile(percentiles: list, scores: list, metric_name: str, file_name=None, single_plot=True):
    """   
        Args:
            precentiles:
                 e.g. [50, 70, 100]
            scores: 
                each element in the list should be a list with the scores
            metric_name:
                y-axis label
            file_name (optional):
                for saving the figure
    """
    if single_plot:
        plt.figure(figsize=(7, 5))
    
    for score in scores:
        plt.plot(percentiles[:], score.data, color=score.color,
                 marker='o', markersize=5, linewidth=2, label=score.model_name)
    
    plt.xticks(list(plt.xticks()[0])[:-2] + [99])
    plt.xlim(percentiles[0], 100)
    #plt.ylim(0)
    plt.legend()
    plt.grid()
    plt.ylabel(metric_name)
    plt.xlabel('Percentile')
    if file_name is not None:
            plt.savefig(file_name, dpi=300, format='pdf')

    if single_plot:
        plt.show()

def plot_histogram(target, data,
                          color='red',
                          model_name='Model'):

    linewidth = 2
    h, b =np.histogram(data.flatten(), bins=np.arange(1, target.max()))
    plt.plot(h/h.sum(), label=f'{model_name}', color=color, linewidth=linewidth)


def plot_precipitation_frequencies(target: np.ndarray, results: list,
                                   units='[?]',
                                   file_name=None,
                                   log_plot=True,
                                   x_max=None,
                                   single_plot=True,
                                   legend_position=None):

    if single_plot:
        plt.figure(figsize =(8,5), dpi=90)

    for r in results:
        plot_histogram(target, r.data, color=r.color, model_name=r.model_name)

    plt.legend(loc=legend_position)
    if log_plot:
        plt.yscale('log')

    if x_max is not None:
        plt.xlim(0, x_max)

    plt.xlabel(f'Rainfall {units}')
    plt.ylabel('Relative frequency')
    plt.grid()

    if file_name is not None:
        plt.savefig(file_name, dpi=300, format='pdf')

    if single_plot:
        plt.show()

@dataclass
class PlotData:
    data: np.ndarray
    color: str
    model_name: str



def plot_accuracy(prediction, baseline,  target, baseline_name='', title='', metric='MSE',
                  file_name=None):
    if metric == 'MSE':
        nn_err = ((prediction - target)**2).mean(axis=(1,2)) 
        baseline_err = ((baseline - target)**2).mean(axis=(1,2)) 

    if metric == 'RMSE':
        nn_err = np.sqrt(((prediction - target)**2).mean(axis=(1,2)))
        baseline_err = np.sqrt(((baseline - target)**2).mean(axis=(1,2)))

    nn_ts_err = np.mean(nn_err)
    baseline_ts_err = np.mean(baseline_err)
    f, ax = plt.subplots(1, 1, figsize=(15, 3), dpi = 80)
    ax.plot(nn_err, label=f'ANN: mean {metric} = {nn_ts_err:.3f}', color='brown') 
    ax.plot(baseline_err, label=f'{baseline_name}: mean {metric} = {baseline_ts_err:.3f}',
            color='tab:blue') 
    ax.set_title(f'{title}') 
    ax.set_ylabel(metric) 
    ax.set_xlabel('JJA days/3hours') 
    plt.legend()
    plt.grid()
    if file_name is not None:
        plt.savefig(file_name, dpi=300, format='pdf')
    plt.show()
    
    
def plot_precipitation_mean(prediction, baseline, target, title, file_name=None):

    prediction_mean = prediction.mean(axis=(1,2))
    baseline_mean = baseline.mean(axis=(1,2))
    target_mean = target.mean(axis=(1,2))

    f, ax = plt.subplots(1, 1, figsize=(15, 3), dpi = 80)

    ax.plot(target_mean, label=f'Target, mean={np.mean(target_mean):.2f}', color='k', linewidth=1.5) 

    ax.plot(baseline_mean, label=f'Baseline, mean={np.mean(baseline_mean):.2f}',
            linewidth=1.5, color='tab:blue') 
    ax.plot(prediction_mean, label=f'ANN, mean={np.mean(prediction_mean):.2f}',
            linewidth=1.5, color='brown') 

    ax.set_title(f'{title}') 
    ax.set_ylabel('Mean') 
    ax.set_xlabel('JJA days/3hours') 
    plt.legend()
    plt.grid()
    if file_name is not None:
        plt.savefig(file_name, dpi=300, format='pdf')
    plt.show()

    
def plot_single_frames(num_frames, data, norm=None,
                       min_cutoff=0, max_cutoff=150, title='DNN',
                       plot_map=False, file_name=None):
    from mpl_toolkits.basemap import Basemap
    plt.figure(figsize=(16,4), dpi=90)
    color_map = 'viridis_r'

    if norm is None:
        norm = data

    if plot_map:
        ds = xr.open_dataset('/path/to/training_dataset', chunks={'time': 92})
        lon_coords = ds.longitude
        lat_coords = ds.latitude

    for i in range(num_frames):

        #cbar = plt.colorbar(cs, fraction=0.046, pad=0.04, label='')
        plt.title(title)
        if plot_map:
            m = Basemap(llcrnrlon=lon_coords[0], llcrnrlat=lat_coords[0],
                  urcrnrlon=lon_coords[-1], urcrnrlat=lat_coords[-1],
                  projection='mill', lon_0=89, lat_0=20, resolution='c')

            m.drawparallels(np.arange(-90, 90, int(len(lat_coords)/10)),
                labels=[1, 0, 0, 0], linewidth=.5)
            m.drawmeridians(np.arange(0, 360, int(len(lon_coords)/16)),
                labels=[0, 0, 0, 1], linewidth=.5)
  
            m.drawcoastlines()
            Lon, Lat = np.meshgrid(lon_coords, lat_coords)
            x, y = m(Lon, Lat)
            m.pcolormesh(x, y, data[i], vmin=norm[i].min()+min_cutoff,  vmax=norm[i].max()-max_cutoff, cmap=color_map)
            plt.colorbar(fraction=0.046, pad=0.04,  extend='both', label=r'Precipitation [mm/3hr]')
        else:
            cs = plt.imshow(data[i], vmin=norm[i].min()+min_cutoff,  vmax=norm[i].max()-max_cutoff, origin='lower', cmap=color_map)            

        if file_name is not None:
            plt.savefig(file_name, dpi=300, format='pdf', bbox_inches = 'tight')

        plt.show()



