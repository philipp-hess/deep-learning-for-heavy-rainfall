import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from src.evaluation_utils import corr, me, rmse
from src.inference import Mask


class GeographicValidation:
    """ Computes grid-cell wise evalutation metrics for ANN model prediction, physical baseline
        and linear model and plots the results. """

    def __init__(self, lats, lons,
                 resolution=0.5,
                 nn_title='DNN',
                 baseline_title='IFS',
                 linear_title='Linear',
                 mask_threshold=None,
                 clean_threshold=None,
                 show_coordinates=True
                 ):
        """
            Args:
                lats (xr.DataArray): shape (n_lats)
                lons (xr.DataArray): shape (n_lons)
                baseline_title (str):
                    name of baseline model
                metrics_list (list): 
                    List of metric strings, e.g. [Mean, RMSE, MASE, Correlation]
                linear_model (np.ndarray): shape (n_time, n_lats, n_lons)
                    linear baseline [optional]
             
        """

        self.lats = lats
        self.lons = lons

        self.nn_title = nn_title
        self.baseline_title = baseline_title
        self.linear_title = linear_title

        self.mask_threshold = mask_threshold
        self.show_coordinates = show_coordinates
        self.clean_threshold = clean_threshold
        self.metrics_list = None
        self.resolution = resolution
        self.file_name = None



    def compute_metrics(self, metrics_list, nn_prediction, baseline, target,
                        linear_model=None,
                        mask_threshold=None,
                        clean_threshold=None,
                        verbose=False):

        self.linear_model = linear_model

        n_lats = target.shape[1]
        n_lons = target.shape[2]

        nn_results = {}
        baseline_results = {}
        linear_results = {}

        for metric in metrics_list:
            nn_results[metric] = np.zeros((n_lats, n_lons))
            baseline_results[metric] = np.zeros((n_lats, n_lons))
            if self.linear_model is not None:
                linear_results[metric] = np.zeros((n_lats, n_lons))

        if mask_threshold is not None:
            target_reference = target.mean(axis=0)
            nn_prediction = np.where(target_reference > mask_threshold, nn_prediction, 0)
            baseline = np.where(target_reference > mask_threshold, baseline, 0)
            target = np.where(target_reference > mask_threshold, target, 0)
            if self.linear_model is not None:
                linear_model = np.where(target_reference > mask_threshold, linear_model, 0)

        if clean_threshold is not None:
            nn_prediction = np.where(target > clean_threshold, nn_prediction, 0)
            baseline = np.where(target > clean_threshold, baseline, 0)
            if self.linear_model is not None:
                linear_model = np.where(target > clean_threshold, linear_model, 0)
            target = np.where(target > clean_threshold, target, 0)
            
        for metric in metrics_list:
            if metric == 'RMSE': 
                func = rmse

            if metric == 'Bias': 
                func = me

            if metric == 'Correlation': 
                #func = spearmanr
                func = corr

            nn_results[metric] = func(nn_prediction, target)
            baseline_results[metric] = func(baseline, target)

            # spearmanr seems to compute NaNs for time series with many 0s
            if np.isnan(np.sum(nn_results[metric])):
                nn_results[metric]= np.nan_to_num(nn_results[metric], 0)

            if np.isnan(np.sum(baseline_results[metric])):
                baseline_results[metric]= np.nan_to_num(baseline_results[metric], 0)

            if self.linear_model is not None:
                if np.isnan(np.sum(linear_results[metric])):
                    linear_results[metric]= np.nan_to_num(linear_results[metric], 0)

       
            
        return nn_results, baseline_results, linear_results


    def add_subplot(self, data, title, ax, vmin, vmax, cmap, counter, 
                    parallels_label=True, meridians_label=True, alpha=0.6,
                    mask=None):

            m, x, y = self.get_basemap(self.lats, self.lons,
                                       orography=self.orography,
                                       parallels_label=parallels_label,
                                       meridians_label=meridians_label,
                                       ax=ax)

            cs = ax.pcolormesh(x, y, data, vmin=vmin, vmax=vmax,
                               alpha=alpha, cmap=cmap,
                               linewidth=0, antialiased=True, rasterized=True, edgecolors='face')

        
            if mask is not None:
                ax.contourf(x,y, mask, hatches=['///', None],  vmin=-1, vmax=-1, alpha=1., cmap='Greys')

            if title is not None:
                ax.set_title(f'{title}')

            counter += 1
            return cs, counter

    def plot_overlap(self, metric, data_1, data_2,
                    configs=None, single_plot=True):

        if configs is None:
            configs = self.get_default_configs()
        self.orography = None


        data = np.where((data_1 <1) | (data_2 <1), None, 1)

        land_sea_mask = Mask(data_1)
        land_sea_mask.create_numpy_mask(tropics_max_latitude=None)
        data_1_masked = land_sea_mask.apply()

        land_sea_mask = Mask(data_2)
        land_sea_mask.create_numpy_mask(tropics_max_latitude=None)
        data_2_masked = land_sea_mask.apply()

        data_masked = np.where((data_1_masked <1) | (data_2_masked <1), None, 1)

        plt.rc('font', size=12)
        plt.rc('legend', fontsize=12)
        
        ax = plt.gca()
        m, x, y = self.get_basemap(self.lats, self.lons,
                                       orography=self.orography,
                                       parallels_label=True,
                                       meridians_label=True,
                                       draw_countries=False,
                                       ax=ax)

        data_masked = data_masked.reshape(data_masked.shape[1], data_masked.shape[2])
        m.contourf(x,y, data, vmin=-1, vmax=1, alpha=0.50, cmap='Greys')
        m.contourf(x,y, data_masked, hatches=['///', '///'], vmin=-1, vmax=1, alpha=0.1, cmap='Greys')

    def get_basemap(self, lats, lons, orography=None, parallels_label=True, meridians_label=True, ax=None):
        """
        Computes the basemap for geographic plotting.

        Args:
            lats (xr.DataArray): shape (n_lats)
            lons (xr.DataArray): shape (n_lons)
            orography: (xr.DataArray): shape (n_lats, n_lons)

        Returns:
            m (basemap)
            x: x coordinates 
            y: y coordinates 
        """
        from mpl_toolkits.basemap import Basemap
        lons = lons.values
        lats = lats.values

        m = Basemap(llcrnrlon=lons[0], llcrnrlat=lats[0],
                    urcrnrlon=lons[-1], urcrnrlat=lats[-1],
                    projection='mill',lon_0=0,resolution='i',
                    ax=ax)

        m.drawcoastlines(color='k')
        m.drawcountries(color='k')

        par = m.drawparallels(np.arange(-90, 90, 30),
                        labels=[parallels_label, 0, 0, 0], linewidth=1.0)

     
        merid = m.drawmeridians([-120, -60, 0, 60, 120],
                        labels=[0, 0, 0, meridians_label], linewidth=1.0)

        Lon, Lat = np.meshgrid(lons, lats)
        x, y = m(Lon, Lat)
        if orography is not None:
            m.contourf(x, y, orography, 15, alpha=1.0, cmap='Greys')
        return m, x, y


    def get_default_configs(self):
        configs = {
                    'RMSE': {
                        'cmap': 'YlOrRd',
                        'cbar_title': f'RMSE [mm/day]',
                        'alpha': 0.6,
                        'vmin': 2.5,
                        'vmax': 13,
                        'cbar_extend': 'both',
                        'linear_title': 'Linear',
                        'dnn_title': 'DNN',
                        'baseline_title': 'Baseline'
                        },

                    'Bias': {
                        'cmap': 'RdBu',
                        'cbar_title': f'ME [mm/day]',
                        'alpha': 0.6,
                        'vmin': -5.5,
                        'vmax': 5.5, 
                        'cbar_extend': 'both',
                        'linear_title': 'Linear',
                        'dnn_title': 'DNN',
                        'baseline_title': 'Baseline'
                        },

                    'Correlation': {
                        'cmap': 'viridis',
                        'cbar_title': f'Correlation',
                        'alpha': 0.6,
                        'vmin': 0,
                        'vmax': 0.65,
                        'cbar_extend': 'both',
                        'linear_title': 'Linear',
                        'dnn_title': 'DNN',
                        'baseline_title': 'Baseline'
                        },

                }
        return configs


    def plot_single(self, metric, data, target,
                    configs=None, single_plot=True):


        if configs is None:
            configs = self.get_default_configs()

        if self.mask_threshold is not None:
            mask = np.where(target > self.mask_threshold, None, 1)
        else: 
            mask = None

        meridians_label = True
        parallel_label = True

        config = configs[metric]

        counter = 0
        if single_plot:
            figure_size=(12, 6)
            fig = plt.figure(figsize=figure_size, dpi=80, constrained_layout=True)

        plt.rc('font', size=12)
        plt.rc('legend', fontsize=12)
        
        ax = plt.gca()

        cs, counter = self.add_subplot(data, config['title'], ax,
                                       config['vmin'], config['vmax'], config['cmap'], counter,
                                       parallels_label=parallel_label, meridians_label=meridians_label,
                                       alpha=config['alpha'], mask=mask)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.15)
        cbar = plt.colorbar(cs, cax=cax, extend=config['cbar_extend'], label=config['cbar_title'])
        cbar.solids.set(alpha=1)
        cs.set_edgecolor("face")

        if single_plot:
            plt.show()

    def plot(self, metrics_list, nn_prediction, baseline, target,
            configs=None, file_name=None, single_plots=False, verbose=False):

        import matplotlib.gridspec as gridspec
        from mpl_toolkits.axes_grid1 import ImageGrid

        if configs is None:
            configs = self.get_default_configs()


        if self.mask_threshold is not None:
            mask = np.where(target.mean(axis=0) > self.mask_threshold, None, 1)
        else: 
            mask = None

        nn_results, baseline_results, linear_results = self.compute_metrics(metrics_list,
                                                                            nn_prediction, baseline, target,
                                                                            linear_model=None,
                                                                            mask_threshold=self.mask_threshold, 
                                                                            clean_threshold=self.clean_threshold,
                                                                            verbose=verbose)

        plt.rc('font', size=12)
        plt.rc('legend', fontsize=12)

        if self.linear_model is not None:
            n_rows = 3*len(metrics_list)
        
        else:
            n_rows = 2*len(metrics_list)

        if not single_plots: 
            figure_size=(12, 6*len(metrics_list))
            fig = plt.figure(figsize=figure_size, dpi=80, constrained_layout=True)
            grid = ImageGrid(fig, 111,     
                        nrows_ncols=(n_rows, 1),
                        axes_pad=0.5,
                        share_all=True,
                        cbar_location="right",
                        cbar_mode="edge",
                        cbar_size="2.5%",
                        cbar_pad=0.15,
                        )

        counter = 0 # indexing the subplots
        for metric in metrics_list:

            if single_plots and self.show_coordinates:
                meridians_label = True
                parallel_label = True
            else:
                meridians_label = False
                parallel_label = False

            config = configs[metric]

            if self.linear_model is not None:
                if single_plots:
                    figure_size=(12, 6)
                    fig = plt.figure(figsize=figure_size, dpi=80, constrained_layout=True)
                    grid = ImageGrid(fig, 111,     
                                nrows_ncols=(1, 1),
                                axes_pad=0.5,
                                share_all=True,
                                cbar_location="right",
                                cbar_mode="edge",
                                cbar_size="2.5%",
                                cbar_pad=0.15,
                                )
                cs, counter = self.add_subplot(linear_results[metric], config['linear_title'], grid[counter],
                                               config['vmin'], config['vmax'], config['cmap'], counter,
                                               parallels_label=parallel_label, meridians_label=meridians_label,
                                               alpha=config['alpha'], mask=mask)
                cbar = grid[counter-1].cax.colorbar(cs, extend=config['cbar_extend'])
                cbar.set_label_text(config['cbar_title'])

                if single_plots:
                    if file_name is not None:
                        file_name_linear = f'{file_name[:-4]}_{metric}_linear.pdf' 
                        print(file_name_linear)
                        plt.savefig(file_name_linear, dpi=100, format='pdf', bbox_inches = 'tight')

            if single_plots:
                counter = 0
                figure_size=(12, 6)
                fig = plt.figure(figsize=figure_size, dpi=80, constrained_layout=True)
                grid = ImageGrid(fig, 111,     
                            nrows_ncols=(1, 1),
                            axes_pad=0.5,
                            share_all=True,
                            cbar_location="right",
                            cbar_mode="edge",
                            cbar_size="2.5%",
                            cbar_pad=0.15,
                            )

            if not single_plots and not self.show_coordinates:
                meridians_label = True

            cs, counter = self.add_subplot(nn_results[metric], config['dnn_title'], grid[counter],
                                           config['vmin'], config['vmax'], config['cmap'], counter,
                                           parallels_label=parallel_label, meridians_label=meridians_label,
                                           alpha=config['alpha'], mask=mask)
            cbar = grid[counter-1].cax.colorbar(cs, extend=config['cbar_extend'])
            cbar.set_label_text(config['cbar_title'])

            if single_plots:
                if file_name is not None:
                    file_name_nn = f'{file_name[:-4]}_{metric}_nn.pdf' 
                    print(file_name_nn)
                    plt.savefig(file_name_nn, dpi=100, format='pdf', bbox_inches = 'tight')
                plt.show()

                counter = 0
                figure_size=(12, 6)
                fig = plt.figure(figsize=figure_size, dpi=80, constrained_layout=True)
                grid = ImageGrid(fig, 111,     
                            nrows_ncols=(1, 1),
                            axes_pad=0.5,
                            share_all=True,
                            cbar_location="right",
                            cbar_mode="edge",
                            cbar_size="2.5%",
                            cbar_pad=0.15,
                            )

            cs, counter = self.add_subplot(baseline_results[metric], config['baseline_title'], grid[counter],
                                           config['vmin'], config['vmax'], config['cmap'], counter,
                                           parallels_label=parallel_label, meridians_label=meridians_label,
                                           alpha=config['alpha'], mask=mask)

            cbar = grid[counter-1].cax.colorbar(cs, extend=config['cbar_extend'])
            cbar.set_label_text(config['cbar_title'])

            if single_plots:
                if file_name is not None:
                    file_name_baseline = f'{file_name[:-4]}_{metric}_baseline.pdf' 
                    print(file_name_baseline)
                    plt.savefig(file_name_baseline, dpi=100, format='pdf', bbox_inches = 'tight')
                plt.show()


        if not single_plots:
            plt.subplots_adjust(wspace=1, hspace=1)
            plt.tight_layout()
            if file_name is not None:
                plt.savefig(file_name, dpi=100, format='pdf', bbox_inches = 'tight')
            plt.show()



def get_basemap(lats, lons, orography=None, parallels_label=True, meridians_label=True, ax=None):
    """
    Computes the basemap for geographic plotting.

    Args:
        lats (xr.DataArray): shape (n_lats)
        lons (xr.DataArray): shape (n_lons)
        orography: (xr.DataArray): shape (n_lats, n_lons)
    
    Returns:
        m (basemap)
        x: x coordinates 
        y: y coordinates 
    """
    from mpl_toolkits.basemap import Basemap
    lons = lons.values
    lats = lats.values
    
    m = Basemap(llcrnrlon=lons[0], llcrnrlat=lats[0],
                urcrnrlon=lons[-1], urcrnrlat=lats[-1],
                projection='mill',lon_0=0,resolution='i',
                ax=ax)
    m.drawcoastlines()
    m.drawcountries()

    par = m.drawparallels(np.arange(-90, 90, int(len(lats)/9)),
                    labels=[parallels_label, 0, 0, 0], linewidth=.5)

    merid = m.drawmeridians(np.arange(0, 360, int(len(lats)/9)),
                    labels=[0, 0, 0, meridians_label], linewidth=.5)
    
    Lon, Lat = np.meshgrid(lons, lats)
    x, y = m(Lon, Lat)
    if orography is not None:
        m.contourf(x, y, orography, 15, alpha=1.0, cmap='Greys')
    return m, x, y




