#!/usr/bin/python
# -*- coding: utf-8 -*-

import config
import pickle

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

from sklearn import preprocessing



def plot_loss(loss, val_loss):
    plt.clf()
    plt.plot(loss, label='loss')
    plt.plot(val_loss, label='val_loss')
    plt.xlabel('epoch')
    plt.ylabel('error')
    plt.legend()
    plt.grid(True)
    plt.show()



class Model:
    def __init__(self):
        pass

    def Build(self):
        raise 'notimplemented'

    def Load(self, fileName):
        raise 'notimplemented'

    def Save(self, fileName):
        raise 'notimplemented'

    def Fit(self, X_trn, y_trn, X_tst, y_tst, plot=False):
        raise 'notimplemented'

    def Evaluate(self, X_tst, y_tst, printErr=True):
        p = self.Predict(X_tst)
        mse = mean_squared_error(y_tst, p)
        mae = mean_absolute_error(y_tst, p)
        if printErr:
            print('MSE = %.06f' % mse)
            print('MAE = %.06f' % mae)
        return mse, mae

    def Predict(self, X):
        raise 'notimplemented'




class KerasModel(Model):
    def Build(self):
      self.model = tf.keras.Sequential([    
        tf.keras.layers.Dense(64, activation='relu'), 
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1),
      ])
      self.model.compile(optimizer=tf.optimizers.Adam(), loss='mean_squared_error')

    def Load(self, fileName):
      self.model = tf.keras.models.load_model(fileName + '.keras')

    def Save(self, fileName):
      self.model.save(fileName + '.keras')

    def Fit(self, X_trn, y_trn, X_tst, y_tst, plot=False):
      early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            verbose=1,
            restore_best_weights=True,
      )

      callbacks = [early_stopping]

      history = self.model.fit(X_trn, y_trn, epochs=1000, validation_data=(X_tst, y_tst), callbacks=callbacks)

      if plot: 
        loss = history.history['loss']
        val_loss =  history.history['val_loss']
        plot_loss(loss, val_loss)


    def Predict(self, X):
      return self.model.predict(X).reshape(-1,1)


class XGBModel(Model):
    def Build(self):
        self.model = XGBRegressor(max_depth=10, n_estimators=1000, objective='reg:squarederror', seed=config.random_state, nthread=12, tree_method='gpu_hist')

    def Load(self, fileName):
        self.Build()
        self.model.load_model(fileName + '.xgb')

    def Save(self, fileName):
        self.model.save_model(fileName + '.xgb')

    def Fit(self, X_trn, y_trn, X_tst, y_tst, plot=False):
        self.model.fit(X_trn, y_trn, eval_metric='rmse', eval_set=[(X_trn, y_trn), (X_tst, y_tst)], verbose=True, early_stopping_rounds=50)
        if plot: 
            results = self.model.evals_result()
            loss = results['validation_0']['rmse']
            val_loss = results['validation_1']['rmse']
            plot_loss(loss, val_loss)

    def Predict(self, X):
        return self.model.predict(X).reshape(-1,1)


class RandomForestModel(Model):
    def Build(self):
        self.model = RandomForestRegressor(n_estimators=100, criterion='mse', random_state=config.random_state, n_jobs=12, verbose=1)

    def Load(self, fileName):
        with open(fileName + '.rf', 'rb') as f:
            self.model = pickle.load(f)

    def Save(self, fileName):
        with open(fileName + '.rf', 'wb') as f:
            pickle.dump(self.model, f)    

    def Fit(self, X_trn, y_trn, X_tst, y_tst, plot=False):
        self.model.fit(X_trn, y_trn)
        if plot: 
            pass

    def Predict(self, X):
        return self.model.predict(X).reshape(-1,1)        


class MultiModel(Model):
    def __init__(self, models):
        self.models = models

    def Build(self):
        for model in self.models:
            model.Build()

    def Load(self, fileName):
        for model in self.models:
            model.Load(fileName)

    def Save(self, fileName):
        for model in self.models:
            model.Save(fileName)

    def Fit(self, X_trn, y_trn, X_tst, y_tst, plot=False):
        for model in self.models:
            model.Fit(X_trn, y_trn, X_tst, y_tst, plot)

    def Evaluate(self, X_tst, y_tst, printErr=True):
        g_mse = 0
        g_mae = 0
        for i, model in enumerate(self.models):
            mse, mae = model.Evaluate(X_tst, y_tst, False)
            if printErr:
                print('MSE[%d] = %.06f' % (i, mse))
                print('MAE[%d] = %.06f' % (i, mae))
            g_mse += mse
            g_mae += mae
        g_mse /= len(self.models)
        g_mae /= len(self.models)
        if printErr:
            print('MSE = %.06f' % g_mse)
            print('MAE = %.06f' % g_mae)
        return g_mse, g_mae

    def Predict(self, X):
        prediction = None
        for model in self.models:
            p = model.Predict(X)
            if prediction is None: prediction = p
            else:
                prediction = prediction + p
        prediction /= len(self.models)
        return prediction
