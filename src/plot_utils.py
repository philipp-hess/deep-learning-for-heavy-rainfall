from dataclasses import dataclass
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
from typing import List

from src.configure import params_from_file
from src.analysis import load_data, relative_improvement
from src.evaluation_geographic import GeographicValidation
from src.evaluation_plots import plot_histogram
from src.inference import Mask


@dataclass
class ScoreData():
    
    percentile: xr.DataArray = None
    ifs: xr.DataArray = None
    dnn: xr.DataArray = None


class PlotSpatialHSSScores():
    
    def __init__(self, data: ScoreData, percentile: float, configs: dict, out_path: str, plot_percentiles=True):
        
        self.data = data 
        self.configs = configs
        self.percentile = percentile
        self.mask_threshold = -1
        self.fname = f'{out_path}geographic_{percentile}th_percentiles_and_hss.png'
        self.in_path = params_from_file('paths')
        self.plot_percentiles = plot_percentiles
        self.figsize = (19,10)
        
        
    def get_coordinates(self):
        
        ds = xr.open_dataset(f'{self.in_path.dataset_path}/{self.in_path.dataset_training}')
        self.lats = ds.latitude
        self.lons = ds.longitude
    
        
    def plot_geographic_percentiles(self):
        
    
        eval = GeographicValidation(self.lats, self.lons,
                                    orography_flag=False,
                                    mask_threshold=None,
                                    clean_threshold=None,
                                    show_coordinates=False)
        data = self.data.percentile
        eval.plot_single('Percentile',  data, data,
                         configs=self.configs, single_plot=False)
        

    def plot_ifs_geographic_hss_scores(self):
        
        eval = GeographicValidation(self.lats, self.lons,
                               orography_flag=False,
                               mask_threshold=self.mask_threshold,
                               clean_threshold=None,
                               show_coordinates=False
                              )
        metric_name = 'HSS'

        data = self.data.ifs
        self.configs['HSS']['title'] = None
        self.configs['HSS']['cbar_title'] = 'IFS HSS'
        eval.plot_single(metric_name,  data, data,
                         configs=self.configs, single_plot=False)

    def plot_hss_geographic_hss_scores(self):
        
        eval = GeographicValidation(self.lats, self.lons,
                               orography_flag=False,
                               mask_threshold=self.mask_threshold,
                               clean_threshold=None,
                               show_coordinates=False
                              )
        metric_name = 'HSS'

        self.configs['HSS']['title'] = None
        self.configs['HSS']['cbar_title'] = 'DNN (WMSE-MS-SSIM) HSS'
        
        data = self.data.dnn
        eval.plot_single(metric_name, data, data,
                 configs=self.configs, single_plot=False)
        
        
    def plot(self):
        
        self.get_coordinates()
        
        plt.rcParams.update({'font.size': 11})
        
        if self.plot_percentiles:
            plt.figure(figsize=self.figsize, dpi=300)
        
            ax1 = plt.subplot(311)
            ax1.annotate("a", ha="center", va="center", size=13,
                     xy=(0.985, 0.945), xycoords=ax1,
                     bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="k", lw=1))
        
            self.plot_geographic_percentiles()
            
        else:
            plt.figure(figsize=self.figsize, dpi=300)
         
        if self.plot_percentiles:
            ax2 = plt.subplot(312)
            annotate = "b"
            
        else:
            ax2 = plt.subplot(211)
            annotate = "a"
            
        ax2.annotate(annotate, ha="center", va="center", size=13,
        xy=(0.985, 0.945), xycoords=ax2,
        bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="k", lw=1))

        self.plot_ifs_geographic_hss_scores()
        
        if self.plot_percentiles:
            ax3 = plt.subplot(313)
            annotate = "c"
            
        else:
            ax3 = plt.subplot(212)
            annotate = "b"
            
        ax3.annotate(annotate, ha="center", va="center", size=13,
        xy=(0.985, 0.945), xycoords=ax3,
        bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="k", lw=1))
        
        self.plot_hss_geographic_hss_scores()
        plt.tight_layout() 
        if self.fname is not None:
            print(self.fname)
            plt.savefig(self.fname, dpi=300, format='png', bbox_inches='tight')
        plt.show()          


class PlotSingleFrames():
    
    def __init__(self, dnn_data, ifs_data, trmm_data,
                 timestamps=['2012-07-16T00', '2013-07-16T00', '2014-07-16T00']):
        
        self.dnn_data = dnn_data
        self.ifs_data = ifs_data
        self.trmm_data = trmm_data
        self.min_precipitation_threshold_in_mm_per_3hours = 0.1
        path = '/path/to/training_dataset'

        self.ds = xr.open_dataset(path, chunks={'time': 1})
        self.lats = self.ds.latitude
        self.lons = self.ds.longitude
        
        self.color_map = 'YlGnBu'
        self.vmin = 0
        self.vmax = 11.5
        self.timestamps = timestamps 
        self.fname = f'/path/to/figures/single_frames.png'

        

    def get_time_indeces(self):

        first_test_set_index = np.where(self.ds.time ==  np.datetime64(f'2012-06-01T00:00:00.000000000'))[0][0]
        index_2012 = np.where(self.ds.time ==  np.datetime64(f'{self.timestamps[0]}:00:00.000000000'))[0][0] - first_test_set_index
        index_2013 = np.where(self.ds.time ==  np.datetime64(f'{self.timestamps[1]}:00:00.000000000'))[0][0] - first_test_set_index
        index_2014 = np.where(self.ds.time ==  np.datetime64(f'{self.timestamps[2]}:00:00.000000000'))[0][0] - first_test_set_index
        
        self.time_idx = [index_2012, index_2013, index_2014]

    def get_data(self):
        self.get_time_indeces() 
        self.dnn_frames = []
        self.ifs_frames = []
        self.trmm_frames = []
        
        for t in self.time_idx:
            self.dnn_frames.append(np.where(self.dnn_data[t] < 
                                   self.min_precipitation_threshold_in_mm_per_3hours,
                                   0, self.dnn_data[t]))
            
            self.ifs_frames.append(np.where(self.ifs_data[t] < 
                                   self.min_precipitation_threshold_in_mm_per_3hours,
                                   0, self.ifs_data[t]))
            
            self.trmm_frames.append(np.where(self.trmm_data[t] < 
                                   self.min_precipitation_threshold_in_mm_per_3hours,
                                   0, self.trmm_data[t]))
            
        
        
    def single_plot(self, data,
                    show_cbar=True,
                    show_latitude_labels=True,
                    show_longitude_labels=True,
                    ):
        m = Basemap(llcrnrlon=self.lons[0], llcrnrlat=self.lats[0],
                    urcrnrlon=self.lons[-1], urcrnrlat=self.lats[-1],
                    projection='merc', lon_0=0, lat_0=20, resolution='c')

        m.drawparallels([-30, 0, 30],
                        labels=[show_latitude_labels, 0, 0, 0], linewidth=.5)
        m.drawmeridians([-120, -60, 0, 60, 120],
                        labels=[0, 0, 0, show_longitude_labels], linewidth=.5)
  
        m.drawcoastlines()
        Lon, Lat = np.meshgrid(self.lons, self.lats)
        x, y = m(Lon, Lat)
        m.pcolormesh(x, y, data, vmin=self.vmin,  vmax=self.vmax, cmap=self.color_map)
        
        if show_cbar:
        
            plt.colorbar(fraction=0.016, pad=0.04,  extend='both', label=r'Precipitation [mm/3hr]')


    def plot(self):
        
        self.get_data()
        
        
        plt.figure(figsize=(15, 6))
        plt.rcParams.update({'font.size': 11})
        for i in range(len(self.time_idx)):
            
            plt.subplot(3,3,1+3*i)
            plt.title(f'IFS, time: {self.timestamps[i]}')
            self.single_plot(self.ifs_frames[i], show_cbar=False,
                             show_latitude_labels=True,
                             show_longitude_labels=i==2)
            
            plt.subplot(3,3,2+3*i)
            plt.title(f'DNN (WMSE-MS-SSIM), time: {self.timestamps[i]}')
            self.single_plot(self.dnn_frames[i], show_cbar=False, 
                             show_latitude_labels=False,
                             show_longitude_labels=i==2)
            
            plt.subplot(3,3,3+3*i)
            plt.title(f'TRMM, time: {self.timestamps[i]}')
            self.single_plot(self.trmm_frames[i], show_cbar=True,
                             show_latitude_labels=False,
                             show_longitude_labels=i==2)

        plt.tight_layout() 
            
        if self.fname is not None:
            print(self.fname)
            plt.savefig(self.fname, dpi=300, format='png')
        plt.show()


class ScoresPerPercentile():
    
    def __init__(self):
        self.heidke_skill_score = {}
        self.f1 = {}
        self.critical_success_index = {}
        self.false_alarm_ratio = {}
        self.probability_of_detection = {}
        
    def set_data(self, data, percentile):
        self.heidke_skill_score[percentile] = data['heidke_skill_score'][0]
        self.f1[percentile] = data['f1'][0]
        self.critical_success_index[percentile] = data['critical_success_index'][0]
        self.false_alarm_ratio[percentile] = data['false_alarm_ratio'][0]
        self.probability_of_detection[percentile] = data['probability_of_detection'][0]
        

@dataclass
class ModelTestSetData():
    
    dnn = None
    dnn_mssim = None
    dnn_weigthed = None
    linear = None
    qm = None
    raw_ifs = None
    

class PlotHistograms():
    
    def __init__(self, path='/path/to/models/'):
        
        self.path = path
        self.model_file_names = {
                  'dnn_weighted': 'dnn_weighted.npy',
                  'dnn_mssim':    'dnn_mssim.npy',
                  'dnn':          'dnn.npy',
                  'qm':           'qm.npy',
                  'linear':       'linear.npy'}
        
    def load_data(self):
        
        fname = f"{self.path}/{self.model_file_names['dnn_weighted']}"
        self.dnn_weighted = np.load(fname)[0]

        fname = f"{self.path}/{self.model_file_names['dnn']}"
        self.dnn_mse = np.load(fname)[0]
        
        fname = f"{self.path}/{self.model_file_names['dnn']}"
        self.trmm = np.load(fname)[1]
        
        fname = f"{self.path}/{self.model_file_names['dnn']}"
        self.ifs = np.load(fname)[2]
        
        fname = f"{self.path}/{self.model_file_names['dnn_mssim']}"
        self.dnn_mssim = np.load(fname)[0]
        
        fname = f"{self.path}/{self.model_file_names['qm']}"
        self.qm = np.load(fname)[0]
        
        fname = f"{self.path}/{self.model_file_names['linear']}"
        self.linear = np.load(fname)[0]


    def mask_data(self):
        
        self.ifs = Mask(self.ifs).apply()
        self.trmm = Mask(self.trmm).apply()
        self.linear = Mask(self.linear).apply()
        self.dnn_mse = Mask(self.dnn_mse).apply()
        self.dnn_mssim = Mask(self.dnn_mssim).apply()
        self.dnn_weighted = Mask(self.dnn_weighted).apply()
        self.qm = Mask(self.qm).apply()


    def plot(self,
             log_plot=True,
             show_plot=True,
             x_max=None,
             fname=None,
             legend_position='upper center',
             masked=False
             ):
        
        plot_histogram(self.trmm, self.ifs, model_name='IFS', color='tab:blue')
        plot_histogram(self.trmm, self.trmm, model_name='TRMM', color='black')
        plot_histogram(self.trmm, self.linear, model_name='Ridge regr.', color='tab:orange')
        plot_histogram(self.trmm, self.dnn_mse, model_name='DNN (MSE)', color='tab:green')
        plot_histogram(self.trmm, self.dnn_mssim, model_name='DNN (MS-SSIM)', color='tab:purple')
        plot_histogram(self.trmm, self.dnn_weighted, model_name='DNN (WMSE-MS-SSIM)', color='tab:red')
        plot_histogram(self.trmm, self.qm, model_name='Quantile map.', color='tab:brown')

        plt.legend(loc=legend_position)
        
        if log_plot:
            plt.yscale('log')

        if x_max is not None:
            plt.xlim(0, x_max)

        plt.xlabel(f'Rainfall [mm/3h]')
        plt.ylabel('Relative frequency')
        plt.grid()

        if fname is not None:
            plt.savefig(fname, dpi=300, format='pdf')

        if show_plot:
            plt.show()


class PlotCategoricalScores():
    
    def __init__(self, in_path: str, percentiles: List[str], model_names: List[str]):
        
        self.in_path = in_path
        self.percentiles = percentiles
        self.model_names = model_names
        self.models = ModelTestSetData()
        self.scores_per_percentile = ScoresPerPercentile
        
        
    def load_results(self):
        for name in self.model_names:
            scores_precentile = self.scores_per_percentile()
            if name == 'dnn_mssim': tmp = scores_precentile
            for percentile in self.percentiles:
                fname = f"{self.in_path}/{name}_percentile_{percentile}"
                data = load_data(fname)
                
                scores_precentile.set_data(data, percentile)
            
            vars(self.models)[name] = scores_precentile
                
                
    def get_results(self):
        return self.models


    def single_plot(self, metric_name: str, file_name=None):
        plt.figure(figsize=(7, 5))
        self.plot(metric_name)
        if file_name is not None:
            plt.savefig(file_name, dpi=300, format='pdf')
        plt.show()        
        
        
    def plot(self, metric_name: str):
        
        colors = {'dnn_weighted': 'tab:red',
                  'dnn_mssim': 'tab:purple',
                  'dnn': 'tab:green',
                  'ifs': 'tab:blue',
                  'linear': 'tab:orange',
                  'qm': 'tab:brown'}

        label = {'dnn_weighted': 'DNN (WMSE-MS-SSIM)',
                 'dnn_mssim': 'DNN (MS-SSIM)',
                 'dnn': 'DNN (MSE)',
                 'ifs': 'IFS',
                 'linear': 'Ridge regr.',
                 'qm': 'Quantile map.'}


        metric_name_dict = {'heidke_skill_score':   'HSS',
                            'f1': 'F1',
                            'critical_success_index':   'CSI',
                            'false_alarm_ratio': 'FAR',
                            'probability_of_detection': 'POD'
        }
    
        for model in vars(self.models):
            scores = vars(self.models)[model]
            score_dict = sorted(getattr(scores, metric_name).items())

            percentiles, scores = zip(*score_dict)
            percentiles = [float(p) for p in percentiles]
            
            plt.plot(percentiles, scores, color=colors[model],
                     marker='o', markersize=5, linewidth=2, label=label[model])
    
        plt.xticks(list(plt.xticks()[0])[:-2] + [99])
        plt.xlim(75, 100)
        #plt.legend(loc='lower left')
        plt.legend()
        plt.grid()
        plt.ylabel(metric_name_dict[metric_name])
        plt.xlabel('Percentile')


    def plot_relative_hss_improvement(self):

        colors = {'dnn_weighted': 'tab:red',
                  'dnn_mssim': 'tab:purple',
                  'dnn': 'tab:green',
                  'ifs': 'tab:blue',
                  'linear': 'tab:orange',
                  'qm': 'tab:brown'}

        label = {'dnn_weighted': 'DNN (WMSE-MS-SSIM)',
                 'dnn_mssim': 'DNN (MS-SSIM)',
                 'dnn': 'DNN (MSE)',
                 'ifs': 'IFS',
                 'linear': 'Ridge regr.',
                 'qm': 'Quantile map.'}

        metric_name = 'heidke_skill_score'
        scores = vars(self.models)['ifs']
        score_dict = sorted(getattr(scores, metric_name).items())
        percentiles, ifs_scores = zip(*score_dict)
    
        for model in vars(self.models):
            scores = vars(self.models)[model]
            score_dict = sorted(getattr(scores, metric_name).items())

            percentiles, scores = zip(*score_dict)
            percentiles = [float(p) for p in percentiles]
            relative_scores = relative_improvement(scores, ifs_scores) 
            plt.plot(percentiles, relative_scores, color=colors[model],
                     marker='o', markersize=5, linewidth=2, label=label[model])
    
        plt.xticks(list(plt.xticks()[0])[:-2] + [99])
        plt.xlim(75, 100)
        plt.legend()
        plt.grid()
        plt.ylabel('Relative improvement [%]')
        plt.xlabel('Percentile')


    def plot_summary(self, file_name=None):

        plt.figure(figsize=(16,10))
        plt.rcParams.update({'font.size': 11})

        ax1 = plt.subplot(221)

        self.plot('heidke_skill_score')
        ax1.annotate("a", ha="center", va="center", size=13,
                     xy=(0.965, 0.945), xycoords=ax1,
                     bbox=dict(boxstyle="square,pad=0.25", fc="white", ec="k", lw=1))
        plt.ylim(-.07,0.45)

        ax2 = plt.subplot(222)
        plt.ylim(-75,500)
    
        ax2.annotate("b", ha="center", va="center", size=13,
                     xy=(0.965, 0.945), xycoords=ax2,
                     bbox=dict(boxstyle="square,pad=0.25", fc="white", ec="k", lw=1))

        self.plot_relative_hss_improvement()
        
        histplot = PlotHistograms()

        ax1 = plt.subplot(223)
        ax1.annotate("c", ha="center", va="center", size=13,
                     xy=(0.965, 0.945), xycoords=ax1,
                     bbox=dict(boxstyle="square,pad=0.25", fc="white", ec="k", lw=1))
        
        histplot.plot(log_plot=False, show_plot=False, x_max=20)

        ax2 = plt.subplot(224)
        histplot.plot(log_plot=True, show_plot=False, x_max=None)

        ax2.annotate("d", ha="center", va="center", size=13,
                     xy=(0.965, 0.945), xycoords=ax2,
                     bbox=dict(boxstyle="square,pad=0.25", fc="white", ec="k", lw=1))
    

        if file_name is not None:
            plt.savefig(file_name, dpi=300, bbox_inches='tight', format='pdf')
        plt.show()       

