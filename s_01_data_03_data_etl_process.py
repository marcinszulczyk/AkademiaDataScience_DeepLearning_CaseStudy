import hydra
import pandas as pd
import numpy as np
from src.config import ExperimentConfig
from numpy import savetxt
import h5py


from library.loglizer.loglizer import dataloader, preprocessing


@hydra.main(config_path="config", config_name="config")
def my_app_etl_raw(experimentConfig : ExperimentConfig):
    np.random.seed(experimentConfig.models.process.seed)

    struct_log = "{}/{}".format(experimentConfig.data.path.raw_data, experimentConfig.data.file.log_file)
    label_file = "{}/{}".format(experimentConfig.data.path.raw_data, experimentConfig.data.file.anomaly_file)

    (x_train, window_y_train, y_train), (x_test, window_y_test, y_test) = dataloader.load_HDFS(struct_log,
                                                                                                label_file  = label_file,
                                                                                                window      = experimentConfig.data.dataloader.window,
                                                                                                window_size = experimentConfig.data.dataloader.window_size,
                                                                                                train_ratio = experimentConfig.data.dataloader.train_ratio,
                                                                                                split_type  = experimentConfig.data.dataloader.split_type)
    
    x_train.to_hdf("{}/{}".format(experimentConfig.data.path.training_data,        experimentConfig.data.file.train_data_raw), key='x_train', mode='w')
    window_y_train.to_hdf("{}/{}".format(experimentConfig.data.path.training_data, experimentConfig.data.file.train_data_raw), key='window_y_train', mode='a')
    y_train.to_hdf("{}/{}".format(experimentConfig.data.path.training_data,        experimentConfig.data.file.train_data_raw), key='y_train', mode='a')

    x_test.to_hdf("{}/{}".format(experimentConfig.data.path.testing_data,          experimentConfig.data.file.test_data_raw),  key='x_test', mode='w')
    window_y_test.to_hdf("{}/{}".format(experimentConfig.data.path.testing_data,   experimentConfig.data.file.test_data_raw),  key='window_y_test', mode='a')
    y_test.to_hdf("{}/{}".format(experimentConfig.data.path.testing_data,          experimentConfig.data.file.test_data_raw),  key='y_test', mode='a')

    
@hydra.main(config_path="config", config_name="config")
def my_app_etl_process_data(experimentConfig : ExperimentConfig):
    np.random.seed(experimentConfig.models.process.seed)

    x_train = pd.read_hdf("{}/{}".format(experimentConfig.data.path.training_data, experimentConfig.data.file.train_data_raw), key='x_train')
    window_y_train = pd.read_hdf("{}/{}".format(experimentConfig.data.path.training_data, experimentConfig.data.file.train_data_raw), key='window_y_train')
    y_train = pd.read_hdf("{}/{}".format(experimentConfig.data.path.training_data, experimentConfig.data.file.train_data_raw), key='y_train')

    x_test = pd.read_hdf("{}/{}".format(experimentConfig.data.path.testing_data, experimentConfig.data.file.test_data_raw), key='x_test')
    window_y_test = pd.read_hdf("{}/{}".format(experimentConfig.data.path.testing_data, experimentConfig.data.file.test_data_raw), key='window_y_test')
    y_test = pd.read_hdf("{}/{}".format(experimentConfig.data.path.testing_data, experimentConfig.data.file.test_data_raw), key='y_test')


    feature_extractor = preprocessing.Vectorizer()
    train_dataset = feature_extractor.fit_transform(x_train, window_y_train, y_train)
    test_dataset  = feature_extractor.transform(x_test, window_y_test, y_test)

    uniqueValues = pd.concat([window_y_train, window_y_test], keys='y_window').nunique()
    print(uniqueValues)

    h5f = h5py.File("{}/{}".format(experimentConfig.data.path.training_data, experimentConfig.data.file.train_data), 'w')
    h5f.create_dataset('x',         data = train_dataset['x'])
    h5f.create_dataset('y',         data = train_dataset['y'])
    h5f.create_dataset('window_y',  data = train_dataset['window_y'])
    h5f.create_dataset('SessionId', data = train_dataset['SessionId'])
    h5f.create_dataset('window_y_uniqueValues', data = uniqueValues)
    h5f.close()

    h5f = h5py.File("{}/{}".format(experimentConfig.data.path.testing_data, experimentConfig.data.file.test_data), 'w')
    h5f.create_dataset('x',         data = test_dataset['x'])
    h5f.create_dataset('y',         data = test_dataset['y'])
    h5f.create_dataset('window_y',  data = test_dataset['window_y'])
    h5f.create_dataset('SessionId', data = test_dataset['SessionId'])
    h5f.create_dataset('window_y_uniqueValues', data = uniqueValues)
    h5f.close()

    df_validation = x_test.copy()
    df_validation['y_window'] = window_y_test
    df_validation['y'] = y_test

    # ===========================================================================================

    session_true  = np.array_split(df_validation[df_validation['y'] == 1] ['SessionId'].unique(), experimentConfig.data.data_comare.split)
    session_false = np.array_split(df_validation[df_validation['y'] == 0] ['SessionId'].unique(), experimentConfig.data.data_comare.split)
    
    h5f = h5py.File("{}/{}".format(experimentConfig.data.path.testing_data, experimentConfig.data.file.model_compare_data_sort), 'w')
    h5f.create_dataset('window_y_uniqueValues', data = uniqueValues)
    h5f.close()

    h5f = h5py.File("{}/{}".format(experimentConfig.data.path.testing_data, experimentConfig.data.file.model_compare_data_unsort), 'w')
    h5f.create_dataset('window_y_uniqueValues', data = uniqueValues)
    h5f.close()
    for x in range(experimentConfig.data.data_comare.split):
        session = np.concatenate((np.array(session_true[x]), np.array(session_false[x])))
        print('= DATA: ',x,'='*80)
        np.random.shuffle(session)
        sort_data = df_validation[df_validation['SessionId'].isin(session)]
        unsort_data = df_validation.reset_index().set_index('SessionId').loc[session].reset_index().set_index('index')
        key = "data_{}".format(x)
        sort_data['EventSequence'].to_hdf("{}/{}".format(  experimentConfig.data.path.testing_data, experimentConfig.data.file.model_compare_data_sort),  key="{}_EventSequence".format(key), mode='a')
        sort_data['y'].to_hdf("{}/{}".format(              experimentConfig.data.path.testing_data, experimentConfig.data.file.model_compare_data_sort),  key="{}_y".format(key), mode='a')
        sort_data['SessionId'].to_hdf("{}/{}".format(      experimentConfig.data.path.testing_data, experimentConfig.data.file.model_compare_data_sort),  key="{}_SessionId".format(key), mode='a')
        sort_data['y_window'].to_hdf("{}/{}".format(       experimentConfig.data.path.testing_data, experimentConfig.data.file.model_compare_data_sort),  key="{}y_window".format(key), mode='a')

        unsort_data['EventSequence'].to_hdf("{}/{}".format(experimentConfig.data.path.testing_data, experimentConfig.data.file.model_compare_data_unsort),  key="{}_EventSequence".format(key), mode='a')
        unsort_data['y'].to_hdf("{}/{}".format(            experimentConfig.data.path.testing_data, experimentConfig.data.file.model_compare_data_unsort),  key="{}_y".format(key), mode='a')
        unsort_data['SessionId'].to_hdf("{}/{}".format(    experimentConfig.data.path.testing_data, experimentConfig.data.file.model_compare_data_unsort),  key="{}_SessionId".format(key), mode='a')
        unsort_data['y_window'].to_hdf("{}/{}".format(     experimentConfig.data.path.testing_data, experimentConfig.data.file.model_compare_data_unsort),  key="{}y_window".format(key), mode='a')


if __name__ == "__main__":
    my_app_etl_raw()
    my_app_etl_process_data()
