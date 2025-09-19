import os
import numpy as np

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from utils.path import Get_path
from utils.utils import *

def create_model( output_info: list[dict[str, str]], dropout_rate: float = 0.3, learning_rate: float = 0.0015) -> tf.keras.Model:
    '''
    Creating model

    output_info Example
    [{'name': 'GENDER', 'nodes': 2}, {'name': 'AGE', 'nodes': 6}]
    '''
    
    # Input layer
    inputs = Input(shape=(768,), name='input')
    x = Dense(512, activation='relu', name='mlp_1')(inputs)
    x = Dropout(dropout_rate)(x)
    x = Dense(256, activation='relu', name='mlp_2')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(128, activation='relu', name='mlp_3')(x)


    outputs = []
    for info in output_info:
        out = Dense(info['nodes'], activation='softmax', name=info['name'])(x)
        outputs.append(out)
    
    # Model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy')

    return model

def train_model(model: tf.keras.Model, X_train: np.ndarray, y_train_dict: dict[str, np.ndarray], 
                batch_size: int =128, epochs: int =100, validation_split: float =0.2):
    
    '''
    Save trained model and return 
    '''

    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        verbose=1, 
        restore_best_weights=True
    )

    model.fit(
        X_train,
        y_train_dict,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        callbacks=[early_stopping],
        verbose=1
    )

    filepath = os.path.join(Get_path.model_path, "trained_model.pkl")
    save_pickle(model, filepath)

    return model





