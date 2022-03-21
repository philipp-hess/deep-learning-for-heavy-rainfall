import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, precision_score, recall_score
from IPython.display import display, clear_output
from sklearn.metrics import confusion_matrix
import scipy.stats as st


def continuous_to_categorical_with_quantiles(data: np.ndarray, quantiles:list ) -> np.ndarray:
    """ Converts continuous data into binar classes using quantiles 
        Args:
            data: shape [n_time, n_lat, n_lon] 
            quantiles:
                list containing quantiles
        Returns:
            tmp: shape [n_quantiles, n_time*n_lat*n_lon]
                binary data
    """

    shape = data.shape
    tmp = np.zeros((len(quantiles), shape[0], shape[1], shape[2])) 
    for i, quantile in enumerate(quantiles):
        threshold = np.quantile(data, quantile)
        binary = np.where(data > threshold, 1, 0).reshape((shape[0], shape[1], shape[2],-1))
        tmp[i] = binary.squeeze()
    return tmp


def global_thresholds_from_quantiles(data: np.ndarray, quantiles:list) -> list:
    thresholds = [np.quantile(data, quantile) for quantile in quantiles]
    return thresholds


def local_thresholds_from_percentiles(data: np.ndarray, percentile: float, data_min=0) -> np.ndarray: 
    n_lat = data.shape[1]
    n_lon = data.shape[2]
    threshold_map = np.zeros((n_lat, n_lon))
    for lat in range(n_lat):
        for lon in range(n_lon):
            tmp = data[:, lat, lon]
            threshold = st.scoreatpercentile(tmp[tmp>data_min], percentile)
            if not np.isnan(threshold):
                threshold_map[lat, lon] = threshold
    return threshold_map


def get_threshold_mask(data: np.ndarray, percentile: float, data_min=0) -> np.ndarray: 
    n_lat = data.shape[1]
    n_lon = data.shape[2]
    mask = np.zeros((n_lat, n_lon))
    for lat in range(n_lat):
        for lon in range(n_lon):
            tmp = data[:, lat, lon]
            threshold = st.scoreatpercentile(tmp[tmp>data_min], percentile)
            if np.isnan(threshold):
                mask[lat, lon] = 1
    return mask


def continuous_to_categorical_with_thresholds(data: np.ndarray, thresholds: list) -> np.ndarray:
    """ Converts continuous data into binar classes using thresholds
        Args:
            data: shape [n_time, n_lat, n_lon] 
            quantiles:
                list containing thresholds
        Returns:
            tmp: shape [n_quantiles, n_time*n_lat*n_lon]
                binary data
    
    """

    shape = data.shape
    tmp = np.zeros((len(thresholds), shape[0], shape[1], shape[2])) 
    for i, threshold in enumerate(thresholds):
        binary = np.where(data > threshold, 1, 0).reshape((shape[0], shape[1], shape[2],-1))
        tmp[i] = binary.squeeze()
    return tmp


def categorical_evaluation(prediction: np.ndarray, target: np.ndarray, metric_name: str, mask=None) -> pd.DataFrame:
    """
        Evaluates a regression prediction with the F1 score
        on quantile-based categories

        Args:
            prediction: shape [n_classes, X]
            target: shape [n_classes, X]

        X can be any other number of dimensions > 0

        Returns:
            scores (list):
                List with an element per class
    """

    n_classes = prediction.shape[0] 
    prediction = prediction.reshape(n_classes, -1)
    target = target.reshape(n_classes, -1)
    
    scores = []
    for c in range(n_classes):
        forecast_skill = ForecastSkill(prediction[c], target[c])
        forecast_skill.compute_categories(mask=mask)

        scores.append(getattr(forecast_skill, f'get_{metric_name}')())

    return scores

def geographic_categorical_evaluation(prediction: np.ndarray, target: np.ndarray, metric_name: str) -> np.ndarray:
    """
        Evaluates a regression prediction with the F1 score
        on quantile-based categories

        Args:
            prediction: shape [n_classes, n_time, n_lat, n_lon]
            target: shape [n_classes, n_time, n_lat, n_lon]

        Returns:
            scores: shape [n_classes, n_lat, n_lon]
    """

    n_classes = prediction.shape[0] 
    n_lat = prediction.shape[2] 
    n_lon = prediction.shape[3] 
    
    scores = np.zeros((n_classes, n_lat, n_lon))
    for c in range(n_classes):
        for lat in range(n_lat):
            for lon in range(n_lon):                    
                grid_cell_prediction = prediction[c, :, lat, lon]
                grid_cell_target = target[c, :, lat, lon]
                if sum(grid_cell_prediction) == 0 and sum(grid_cell_target) == 0:
                    scores[c, lat, lon] = -999
                else:
                    forecast_skill = ForecastSkill(prediction[c, :, lat, lon], target[c, :, lat, lon])
                    forecast_skill.compute_categories()
                    scores[c, lat, lon] = getattr(forecast_skill, f'get_{metric_name}')()
                print(f'Progress {int((lat * lon)/(n_lat*n_lon)*100):2d}%')
                clear_output(wait=True)
    return scores

class ForecastSkill:
    
    """ A collection of categorical forecast skill metrics """
    
    def __init__(self, prediction, target):
    
        self.prediction = prediction
        self.target = target
        self.true_positive = 0
        self.false_positive = 0
        self.false_negative = 0
        self.true_negative = 0


    def compute_categories(self, mask=None):
        
        self.target = self.target.flatten().astype('int')
        self.prediction = self.prediction.flatten().astype('int')

        if mask is not None:
            mask = mask.flatten()
            indices_to_remove = np.where(mask==1)
            self.target = np.delete(self.target, indices_to_remove)
            self.prediction = np.delete(self.prediction, indices_to_remove)

        categories = confusion_matrix(self.target, self.prediction)
        self.true_negative, self.false_positive, self.false_negative, self.true_positive = categories.ravel()


    def print_category_sums(self):
        total = self.target.size
        print(f'tp: {self.true_positive/total*100:2.3f}')
        print(f'fp: {self.false_positive/total*100:2.3f}')
        print(f'fn: {self.false_negative/total*100:2.3f}')
        print(f'tn: {self.true_negative/total*100:2.3f}')


    def get_category_sums(self):
        return self.true_positive, self.false_positive, self.false_negative, self.true_negative
        
    
    def get_heidke_skill_score(self) -> float:
        
        tp = self.true_positive 
        fp = self.false_positive
        fn = self.false_negative 
        tn = self.true_negative 
            
        nominator = 2*(tp*tn - fp*fn)
        denominator = ((tp + fn)*(fn + tn) + (tp + fp)*(fp + tn))
        if denominator > 0:
            return nominator/denominator
        else:
            raise ValueError('devision by zero')

        
    def get_critical_success_index(self) -> float:
        
        hits = self.true_positive 
        false_alarms = self.false_positive
        misses = self.false_negative 
        
        nominator = hits
        denominator = hits + misses + false_alarms
        
        if denominator > 0:
            return nominator/denominator
        else:
            raise ValueError('devision by zero')
            
            
    def get_false_alarm_ratio(self) -> float:
        
        hits = self.true_positive 
        false_alarms = self.false_positive
        
        nominator = false_alarms
        denominator = hits + false_alarms
        
        if denominator > 0:
            return nominator/denominator
        else:
            raise ValueError('devision by zero')
            
            
    def get_probability_of_detection(self) -> float:
        
        hits = self.true_positive 
        misses = self.false_negative 
        
        nominator = hits
        denominator = hits + misses
        
        if denominator > 0:
            return nominator/denominator
        else:
            raise ValueError('devision by zero')
            
            
    def get_f1(self) -> float:
        return f1_score(self.target, self.prediction, average='binary')
    
    
    def get_recall(self) -> float:
        return recall_score(self.target, self.prediction, average='binary')
    
    
    def get_precision(self) -> float:
        return precision_score(self.target, self.prediction, average='binary')


def rmse(output, target):
    return np.sqrt(((output-target)**2).mean(axis=0))


def me(output, target):
    return (output-target).mean(axis=0)


def corr(output, target):
    result =  np.zeros((output.shape[1], output.shape[2]))
    for i in range(output.shape[1]):
        for j in range(output.shape[2]):  
            result[i,j] = spearmanr(output[:,i,j], target[:,i,j])[0]
    return result

