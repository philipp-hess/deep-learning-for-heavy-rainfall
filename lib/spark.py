from pyspark.sql.functions import col
from pyspark.sql import  SQLContext
from pyspark import SparkContext
import pandas as pd

class ModelData:
    def __init__(self):
    
        sc = SparkContext.getOrCreate()
        self.sqlc = SQLContext(sc)


    def import_data(self, in_path):
        self.in_path = in_path 
        self.files = self.sqlc.read.json(in_path+'/*.json', multiLine=True)

        
    def get_training_data(self, model_name,  start_date, end_date):
        """ dates should have format yyyy/mm/dd """
        
        df = self.files.where(col('model_name') == model_name)

        attributes = ['id', 'min validation loss', 'date', 'n_epochs']
        model_stats = df.select([c for c in df.columns if c in attributes])

        model_stats = model_stats.toPandas()
        
        if start_date is not None and end_date is not None:
            mask = (model_stats['date'] > start_date) & (model_stats['date'] <= end_date)    
            model_stats = model_stats.loc[mask]
        
        return model_stats

        
    def get_training_df(self, model_name):
        df = self.files.where(col('model_name') == model_name)
        return df

      
    def get_model_hyperparams(self, model_id):
        model_id = model_id
        df = self.files.where(col('id') == model_id)
        hparams = df.select(col('hyperparameters.*')).toPandas()
        # remove NaN-columns
        hparams = hparams[hparams.columns[~hparams.isnull().all()]]
        return hparams

    def get_model_params(self, model_id):
        model_id = model_id
        df = self.files.where(col('id') == model_id)
        hparams = df.toPandas()
        # remove NaN-columns
        hparams = hparams[hparams.columns[~hparams.isnull().all()]]
        return hparams

    
    def get_training_dataset(self, model_id):
        df = self.files.where(col('id') == model_id)
        path = df.select(col("paths.*")).select(col("dataset_path")).collect()[0]["dataset_path"]
        file_name_training = df.select(col("paths.*")).select(col("dataset_training")).collect()[0]["dataset_training"]
        file_name_test = df.select(col("paths.*")).select(col("dataset_test")).collect()[0]["dataset_test"]
        test_file = f'{path}/{file_name_test}'
        training_file = f'{path}/{file_name_training}'
        return training_file, test_file


    def get_training_options(self, model_id):
        self.file = self.sqlc.read.json(self.in_path+f'/*{model_id}*.json', multiLine=True)
        df = self.file.where(col('id') == model_id)
        training_options_dict = df.select(col("training_options.*")).toPandas().to_dict('records')[0]
        features = df.select(col("training_options.*")).select(col("features.*")).toPandas().to_dict('records')[0]

        del training_options_dict['features']
        training_options_dict['features'] = features

        return training_options_dict


    def get_training_paths(self, model_id):
        df = self.files.where(col('id') == model_id)
        return df.select(col("paths.*")).toPandas().to_dict('records')[0]


    def get_validation_min_model_id(self, model_stats):
        index = model_stats['min validation loss'].idxmin()
        model_id = model_stats.loc[index]['id']
        return model_id
