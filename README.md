# Deep Learning for Improving Numerical Weather Prediction of Heavy Rainfall

## Description
This repository contains the code for data pre-processing, model training and evaluation used in the study

  - "Deep Learning for Improving Numerical Weather Prediction of Heavy Rainfall"


## Requirements
The dependencies can be installed via pip:

```
python3 -m venv <environment_name>
source <environment_name>/bin/activate
pip install -r requirements.txt
```

## Data

- The TRMM (TMPA) data can be downloaded from [NASA GES DISC](https://disc.gsfc.nasa.gov/datasets/TRMM_3B42_7/summary).

- The IFS data is available at the [Climate Data Store](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels?tab=form).

## Usage
### Pre-processing:

Create dataset with

```
python3 preproc/collect_training_data.py
```


Prepare the data for training with
1. Define the parameters and file paths in params.json
2. run: 

```
python3 preproc/build_training_dataset.py
```

### Training:
1. Define the parameters and file paths in params.json
2. run: 

```
python3 main.py --params.json
```

### Evaluation:

The quantile mapping method can be run with
```
python3 src/quantile_mapping.py
```

To evaluate the results use the notebook:

- `notebooks/evaluate-models.ipynb`

To reproduce the figures use:
- `notebooks/create-categorical-plots.ipynb`
- `notebooks/create-spatial-plots.ipynb`
