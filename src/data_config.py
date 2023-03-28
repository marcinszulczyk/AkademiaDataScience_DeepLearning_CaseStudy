from dataclasses import dataclass

@dataclass
class Loglizer:
    path    : str

@dataclass
class File:
    src_path_log            : str
    src_path_anomaly        : str
    path_raw_data           : str
    path_process_data       : str
    path_testing_data       : str
    path_training_data      : str

    log_file                : str
    anomaly_file            : str

    train_data_raw          : str
    test_data_raw           : str
    train_data              : str
    test_data               : str
    model_compare_data_sort     : str
    model_compare_data_unsort     : str

    model_compare_data_01   : str
    model_compare_data_02   : str
    model_compare_data_03   : str
    model_compare_data_04   : str
    model_compare_data_05   : str
    model_compare_data_06   : str
    model_compare_data_07   : str
    model_compare_data_08   : str
    model_compare_data_09   : str
    model_compare_data_10   : str

@dataclass
class Data:
    train_ratio  : float
    window_size  : int
    seed         : int

@dataclass 
class DataConfig:
    loglizer : Loglizer
    file     : File
    data     : Data
