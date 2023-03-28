from dataclasses import dataclass

@dataclass
class Path:
    src_log_file            : str
    src_anaomaly_file            : str
    src_path_log            : str
    src_path_anomaly        : str
    raw_data                : str
    path_process_data       : str
    testing_data            : str
    training_data           : str
    neptune                 : str

@dataclass
class File:
    log_file                : str
    anomaly_file            : str
    train_data              : str
    test_data              : str
    testing_data_raw            : str
    training_data_raw           : str
    neptune                 : str
    model_compare_data_sort : str
    model_compare_data_unsort : str

@dataclass
class Dataloader:
  window      : str
  window_size : int
  train_ratio : float
  split_type  : str

@dataclass
class Data_comare:
    split : int

@dataclass
class Data:
    path        : Path
    file        : File
    dataloader  : Dataloader
    data_comare : Data_comare

@dataclass
class Process:
    name             : str
    epochs           : int
    batch_size       : int
    validation_split : float
    patience         : int
    seed             : int

@dataclass
class Models:
    name      : str
    optimizer : str
    loss      : str
    layer     : list
    path      : str
    file      : str

@dataclass 
class ExperimentConfig:
    data      : Data
    models    : Models
    process   : Process
