import sys

from src.modelTemplate import ModelTemplate

class Model(ModelTemplate):
    def model_get(self):

        layer_list = [(k, v) for k, v in self.model_layers.items()]

        initializer = self.krs.initializers.Ones()
        self.logger.info ("")
        self.logger.info ("Model get 2")
        self.logger.info ("-" * 80)
        self.logger.info ("input_dim: " , self.input_dim)
        self.logger.info ("output_dim: " , self.output_dim)
        self.logger.info ("input_length: " , self.input_length)

        self.model = self.krs.Sequential([
            self.krs.layers.Embedding(input_dim=self.input_dim, output_dim=self.output_dim, input_length=self.input_length),
            self.krs.layers.Bidirectional(self.krs.layers.LSTM(layer_list[0][1]['units'], return_sequences=False, kernel_initializer=initializer)),
            self.krs.layers.Dropout(layer_list[1][1]['dropout']),
            self.krs.layers.Dense(layer_list[2][1]['units'], activation=layer_list[2][1]['activation'], kernel_initializer=initializer),
            self.krs.layers.Dropout(layer_list[3][1]['dropout']),
            self.krs.layers.Dense(layer_list[4][1]['units'], activation=layer_list[4][1]['activation'], kernel_initializer=initializer),
            self.krs.layers.Dropout(layer_list[5][1]['dropout']),
            self.krs.layers.Dense(self.input_dim, activation='softmax', kernel_initializer=initializer)
            ])    
