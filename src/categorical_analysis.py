from analysis import load_test_data, compute_thresholds, transform_to_categorical_data, compute_scores,\
save_data


"""Script to run the categorical evaluation"""


def main():

    """Parameter: """

    path_dicts = [{'dnn_weighted': '/path/to/models/dnn_weighted.npy'},
                  {'dnn_mssim':    '/path/to/models/dnn_mssim.npy'},
                  {'dnn':          '/path/to/models/dnn.npy'},
                  {'qm':           '/path/to/models/qm.npy'},
                  {'linear':       '/path/to/models/linear.npy'}]

    dataset_path = '/path/to/training_dataset.nc4'
    out_path = '/path/to/analysis/analysis'

    percentiles = [75, 80, 85, 90, 95, 97.5, 99, 99.9]

    min_precipitation_threshold_in_mm_per_3hours = 0.1
    time_period = ('1998', '2008')

    metrics = ['heidke_skill_score',
               'f1',
               'critical_success_index',
               'false_alarm_ratio',
               'probability_of_detection']

    """Compute: """

    for path_dict in path_dicts:

        test_data = load_test_data(path_dict)

        for percentile in percentiles:

            thresholds, masks = compute_thresholds(dataset_path, [percentile],
                                                   min_precipitation_threshold_in_mm_per_3hours,
                                                   time_period=time_period)

            categorical_test_data = transform_to_categorical_data(test_data, thresholds)

            scores = compute_scores(categorical_test_data, metrics, masks)

            for name, data in scores.items():
                fname = f'{out_path}{name}_percentile_{percentile}'
                save_data(data, fname)

if __name__ == '__main__':
    main()