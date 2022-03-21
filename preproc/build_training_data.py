import sys
sys.path.append("..")
from src.dataset import DatasetStandardization
import json

def transform():
    """
    Loads training parameters and splits the netcdf dataset
    for crossvalidation and performs features standardization.

    The location of the source dataset file is specified in ../params.json
    under the "dataset_path" key and the file name with "dataset_training".

    Returns:
        Standardized dataset in paths['dataset_path']/standarized/

    Note:
        Running this script might require large memory
    """


    with open('../params.json') as json_file:
        params = json.load(json_file)
    
    paths = params['paths']

    output_format = paths['input_format']
    training_params = params['training_params']
    hparams = params['hparams']

    feature_dict = training_params['features']
    target_str = training_params['target']

    for flag in ['training', 'validation', 'test']:
        standarized = DatasetStandardization(flag, feature_dict, target_str, training_params, paths)
        standarized.write_to_file(format=output_format)

if __name__ == '__main__':
    transform()