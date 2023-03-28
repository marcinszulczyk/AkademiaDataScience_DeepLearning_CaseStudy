import hydra
import wget
from src.config import ExperimentConfig

@hydra.main(config_path="config", config_name="config")
def my_app(experimentConfig : ExperimentConfig):
    
    wget.download(url = experimentConfig.data.path.src_log_file, 
                                 out = "{}/{}".format(experimentConfig.data.path.raw_data, experimentConfig.data.file.log_file))
    
    wget.download(url = experimentConfig.data.path.src_anaomaly_file, 
                                 out = "{}/{}".format(experimentConfig.data.path.raw_data, experimentConfig.data.file.anomaly_file))

if __name__ == "__main__":
    my_app()
