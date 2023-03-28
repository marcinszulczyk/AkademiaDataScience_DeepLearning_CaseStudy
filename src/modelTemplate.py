import tensorflow as tf
import tensorflow_addons as tfa
from src.config import ExperimentConfig
import h5py
import logging
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import pandas as pd

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.models import load_model

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import numpy as np
import random
import os
import json
import os

import neptune.new as neptune
from neptune.new.integrations.tensorflow_keras import NeptuneCallback

class ModelTemplate():
    def __init__(self, experimentConfig : ExperimentConfig):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

        self.experimentConfig = experimentConfig
        self.neptune_enable = 0
        self.load_neptune_config()

        if self.neptune_enable:
            self.run_neptune = neptune.init_run(project=self.project, api_token=self.api_token)

        self.model_layers  = experimentConfig.models.model.layer
        self.path_grapf    = 'graph'
        self.path_models   = 'models'
        self.path_reports  = 'reports'
        os.mkdir(self.path_grapf)
        os.mkdir(self.path_models)
        os.mkdir(self.path_reports)
        
        self.seed                 = experimentConfig.models.process.seed
        self.epochs               = experimentConfig.models.process.epochs
        self.batch_size           = experimentConfig.models.process.batch_size
        self.validation_split     = experimentConfig.models.process.validation_split
        self.patience             = experimentConfig.models.process.patience

        self.filepatch_model      = experimentConfig.models.model.name + '_' + experimentConfig.models.process.name + '_' + '_epoch-{epoch:04d}_val_loss-{val_loss:.04f}.hdf5'
        self.filepatch_best_model = experimentConfig.models.model.name + '_' + experimentConfig.models.process.name + '_' + '_best_model.hdf5'

        self.filepatch_model      = "{}/{}".format(self.path_models, self.filepatch_model)
        self.filepatch_best_model = "{}/{}".format(self.path_models, self.filepatch_best_model)

        self.logger = logging.getLogger("name")
        self.krs = tf.keras
        
        self.logger.info  ("")
        self.logger.info  ("Init model")
        self.logger.info  ("-" * 80)
        self.logger.info  ("Monel name    : {}".format(self.experimentConfig.models.model.name))
        self.logger.info  ("Monel process : {}".format(self.experimentConfig.models.process.name))
        
        if self.neptune_enable:
            self.run_neptune["parameters"] = {
                "Name"    : self.experimentConfig.models.model.name,
                "Process" : self.experimentConfig.models.process.name,
                "Epochs" : self.experimentConfig.models.process.epochs,
                "Batch_size" : self.experimentConfig.models.process.batch_size,
                "Validation_split" : self.experimentConfig.models.process.validation_split,
                "Patience" : self.experimentConfig.models.process.patience,
                "Seed" : self.experimentConfig.models.process.seed,
            }

    def load_neptune_config(self):
        print (self.experimentConfig.data.path.training_data + " " +self.experimentConfig.data.file.train_data)
        neptune_file = r'{}/{}'.format(self.experimentConfig.data.path.neptune, self.experimentConfig.data.file.neptune)
        if os.path.exists(neptune_file):
            try:
                f = open(neptune_file)
                data = json.load(f)
            except:
                self.logger.error("ERROR")
            finally:
                f.close()
                
            try:
                print(data['project'])
                print(data['api_token'])
                self.project=data['project']
                self.api_token=data['api_token']
                self.neptune_enable = 1
            except:
                self.neptune_enable = 0
                self.project=""
                self.api_token=""
                self.logger.error("ERROR")

    def data_load(self):
        self.logger.info  ("")
        self.logger.info  ("Data load")
        self.logger.info  ("-" * 80)
        self.logger.info  ("Data train: {}/{}".format(self.experimentConfig.data.path.training_data, self.experimentConfig.data.file.train_data))
        self.logger.info  ("Data test : {}/{}".format(self.experimentConfig.data.path.testing_data, self.experimentConfig.data.file.test_data))
    
        h5f = h5py.File("{}/{}".format(self.experimentConfig.data.path.training_data, self.experimentConfig.data.file.train_data),'r')
        self.X_train = h5f['x'][:]
        self.y_train = h5f['y'][:]
        self.window_y_train = h5f['window_y'][:]
        self.SessionId_train = h5f['SessionId'][:]    
        self.window_y_uniqueValues = h5f['window_y_uniqueValues'][()]
        h5f.close()

        h5f = h5py.File("{}/{}".format(self.experimentConfig.data.path.testing_data, self.experimentConfig.data.file.test_data),'r')
        self.X_test = h5f['x'][:]
        self.y_test = h5f['y'][:]
        self.window_y_test = h5f['window_y'][:]
        self.SessionId_test = h5f['SessionId'][:]    
        h5f.close()

        self.input_dim    = self.window_y_uniqueValues + 1
        self.input_length = self.X_train.shape[1]
        self.output_dim   = self.window_y_uniqueValues

    def model_compile(self):
        self.logger.info  ("")
        self.logger.info  ("Model compile")   
        self.logger.info  ("-" * 80)   
        self.logger.info  ("Inpout dim : {}".format(self.input_dim ))   
        self.logger.info  ("Metrics    : F1-score, accuracy")   
        
        self.model.compile(optimizer=self.experimentConfig.models.model.optimizer,
                           loss=self.experimentConfig.models.model.loss,
                           metrics=[tfa.metrics.F1Score(num_classes=self.input_dim), 'accuracy'])

    def model_train(self):        
        os.environ['PYTHONHASHSEED']=str(self.seed)
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        self.logger.info  ("")
        self.logger.info  ("Model train")   
        self.logger.info  ("-" * 80)   
        self.logger.info  ("Seed             : {}".format(self.seed)) 
        self.logger.info  ("Epochs           : {}".format(self.epochs))   
        self.logger.info  ("Batch size       : {}".format(self.batch_size))   
        self.logger.info  ("Validation split : {}".format(self.validation_split))   
        self.logger.info  ("EarlyStopping    : val_loss") 
        self.logger.info  ("Patience         : {}".format(self.patience)) 

        checkpoint_model      = ModelCheckpoint(filepath=self.filepatch_model,      monitor='val_accuracy', mode='max', verbose = 0, save_best_only =True)
        checkpoint_best_model = ModelCheckpoint(filepath=self.filepatch_best_model, monitor='val_accuracy', mode='max', verbose = 1, save_best_only =True)
  
        checkpoint_model      = ModelCheckpoint(filepath=self.filepatch_model,      monitor='val_loss', mode='min', verbose = 0, save_best_only =True)
        checkpoint_best_model = ModelCheckpoint(filepath=self.filepatch_best_model, monitor='val_loss', mode='min', verbose = 1, save_best_only =True)
  
        es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = self.patience)
  
        callbacks = []
        if self.neptune_enable:
            neptune_cbk = NeptuneCallback(run=self.run_neptune)
            callbacks = [es, checkpoint_best_model, checkpoint_model, neptune_cbk]
        else:
            callbacks = [es, checkpoint_best_model, checkpoint_model]
            

        self.history =  self.model.fit(self.X_train,
                                       self.krs.utils.to_categorical(self.window_y_train, self.input_dim),
                                       epochs=self.epochs,
                                       batch_size=self.batch_size,
                                       validation_split=self.validation_split,
                                       callbacks=callbacks)
        
        self.model = load_model(self.filepatch_best_model)

    def check_roc_curve(self, y_true, y_pred):
        from sklearn.metrics import roc_curve
        fpr, tpr, tresh = roc_curve(y_true, y_pred, pos_label = 1)
        roc = pd.DataFrame({'fpr':fpr, 'tpr':tpr})
        fig = go.Figure(
            data=[
                go.Scatter(x=roc['fpr'], y=roc['tpr'],
                    line_color='red',
                    name='Roc Curve'),
                go.Scatter(x=[0,1], y=[0,1],
                    mode='lines',
                    line_dash='dash',
                    line_color='navy')
            ],
            layout=go.Layout(
                xaxis_title = "False Positive Rate",
                yaxis_title = "True Positive Rate",
                title="ROC Curve",
                showlegend=False,
                width=800,
                font_size=12) 
                )
        fig.write_image(r"{}/roc.png".format(self.path_grapf))
        roc_auc = auc(fpr, tpr)
        gini = (2*roc_auc) - 1
        return roc_auc, gini
    
    def plot_hist(self, history):
        hist =pd.DataFrame(history.history)  
        hist['epoch'] = history.epoch

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['accuracy'], name ='accuracy', mode='markers+lines'))
        fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['val_accuracy'], name ='val_accuracy', mode='markers+lines'))
        fig.update_layout(width=1000, height=500, title='accuracy vs. val accuracy', xaxis_title='Epoch', yaxis_title='Accuracy')
        fig.write_image(r"{}/accuracy.png".format(self.path_grapf))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['loss'], name ='loss', mode='markers+lines'))
        fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['val_loss'], name ='val_loss', mode='markers+lines'))
        fig.update_layout(width=1000, height=500, title='loss vs. val loss', xaxis_title='Epoch', yaxis_title='Loss')
        fig.write_image(r"{}/loss.png".format(self.path_grapf))

    def model_predict(self):
        self.logger.info  ("")
        self.logger.info  ("Model Predict")   
        self.logger.info  ("-" * 80)  
        self.model = load_model(self.filepatch_best_model)
        
        self.predict = self.model.predict(self.X_test)

        self.y_anomaly = []
        for idx in range(self.window_y_test.size):
            next_event = self.window_y_test[idx]
            most_probably_events = self.predict[idx].argsort()[::-1]
            top = most_probably_events[:3]
            if next_event not in top:
                self.y_anomaly.append(1)
            else:
                self.y_anomaly.append(0)      

    def plot_confusion_matrix(self, cm):
        cm = cm[::-1]
        cm = pd.DataFrame(cm, columns=['Pred False', 'Pred True'], index=['Y True', 'Y False'])

        fig = ff.create_annotated_heatmap(z=cm.values, x=list(cm.columns), y=list(cm.index),
                                            colorscale='ice', showscale=True, reversescale=True)
        fig.update_layout(width=800, height=800, title='Confusion Matrix', font_size=12)  
        fig.write_image(r"{}/confusion_matrix.png".format(self.path_grapf))
 
    def model_get(self):
        self.logger.info  ("")
        self.logger.info  ("Model get")
        self.logger.info  ("-" * 80)  

    def report_create(self):
        self.logger.info  ("")
        self.logger.info  ("Report create")
        self.logger.info  ("-" * 80)  
        
        self.plot_hist(self.history)

        self.cf = confusion_matrix(y_true=self.y_test, y_pred=self.y_anomaly)
        self.plot_confusion_matrix(self.cf)
        roc_auc, gini = self.check_roc_curve(self.y_test, self.y_anomaly)
        data = [["roc", roc_auc], ["gini", gini]]        
        df = pd.DataFrame(data, columns=['Metrics', 'Value'])
        df.to_csv(r"{}\roc_gini.csv".format(self.path_reports))

        clasification_report = classification_report(y_true=self.y_test , y_pred=self.y_anomaly)
        self.logger.info  ("{}".format(clasification_report))
        clasification_report_table = classification_report(y_true=self.y_test , y_pred=self.y_anomaly, output_dict=True) 
        df_classification_report = pd.DataFrame(clasification_report_table).transpose()
        df_classification_report.to_csv(r"{}\clasification_report.csv".format(self.path_reports), index = False) 
        self.logger.info  ("AUC   : {}".format(roc_auc)) 
        self.logger.info  ("GINI  : {}".format(gini)) 

        print (type(df_classification_report))
        print (df_classification_report["precision"][0])
        print (df_classification_report["precision"][1])
        # a = df_classification_report[0]
        # b = df_classification_report[0][0]
        if self.neptune_enable:
            self.run_neptune["parameters"] = {
                "TP": self.cf[0][0],
                "FP": self.cf[0][1],
                "FN": self.cf[1][0],
                "TN": self.cf[1][1],
                "GINI": gini,
                "AUC": roc_auc,
                "Precision_N" : df_classification_report["precision"][0],
                "Precision_P" : df_classification_report["precision"][1],
                "Recall_N" : df_classification_report["recall"][0],
                "Recall_P" : df_classification_report["recall"][1],
                "F1_SCORE_N" : df_classification_report["f1-score"][0],
                "F1_SCORE_P" : df_classification_report["f1-score"][1]
                }
        if self.neptune_enable:
            self.run_neptune["summary/report/clasification_report"].upload(clasification_report)
            self.run_neptune["summary/report/clasification_report_table"].upload(r"{}\clasification_report.csv".format(self.path_reports))
            self.run_neptune["summary/graph/confusion_matrix"].upload(r"{}/confusion_matrix.png".format(self.path_grapf))
            self.run_neptune["summary/graph/roc"].upload(r"{}/roc.png".format(self.path_grapf))

    def model_run(self):
        try:
            self.data_load()
            self.model_get()
            self.model_compile()
            self.model_train()
            self.model_predict()
            self.report_create()
        finally:
            if self.neptune_enable:
                self.run_neptune.stop()

