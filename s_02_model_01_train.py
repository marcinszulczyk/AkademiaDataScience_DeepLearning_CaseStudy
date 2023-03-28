import sys
import hydra
import imp
from src.config import ExperimentConfig

@hydra.main(config_path="config", config_name="config")
def my_app(experimentConfig : ExperimentConfig):    
    patch      =  experimentConfig.models.model.path
    model_name = experimentConfig.models.model.file
    model_path = "{}".format(patch)

    f_p, f_patch, desc = imp.find_module(name = model_name, path=[model_path] )
    module = imp.load_module(model_name,f_p, f_patch, desc)
    a = module.Model(experimentConfig)
    a.model_run()

if __name__ == "__main__":
    my_app()
    