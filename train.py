#!/usr/bin/python
# -*- coding: utf-8 -*-

import config

import pandas as pd
pd.options.display.float_format = "{:.2f}".format


import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error

import seaborn as sns

import models
import pickle

def plot_predictions(true_values, predictions):
  plt.clf()
  ax = plt.axes(aspect='equal')
  plt.scatter(true_values, predictions)
  plt.xlabel('true values')
  plt.ylabel('predictions')
  ax.plot([0, 1], [0, 1], transform=ax.transAxes)
  plt.show()


if __name__ == "__main__":

  predictOnly = False

  save = False
  load = True
  retrain = False
  plot = False


  print('reading...')


  argc = len(sys.argv)
  if argc > 1 and sys.argv[1] == '-train':
    save = True
    load = False
    retrain = True
    plot = False

  if argc > 1 and sys.argv[1] == '-predictonly':
    # if -predictonly is given, just read the whole file and predict the values using the already trained model
    # this file should of course only contain never seen records
    predictOnly = True
    tst = pd.read_csv('flights.csv', delimiter = ',')
    # trn isn't used in that case but it must be non empty
    trn = tst[0:1].copy()
  else:
    trn = pd.read_csv('flights_trn.csv', delimiter = ',')
    tst = pd.read_csv('flights_tst.csv', delimiter = ',')


  #g = sns.pairplot(trn[['duration', 'month', 'CHT1', 'CHT3', 'EGT1', 'EGT3', 'FF', 'OILT', 'MAP', 'RPM', 'OAT']], diag_kind='kde')
  #g.map_lower(sns.kdeplot, levels=4, color=".2")
  #plt.show()


  # the model will predict CHT and EGT temperatures for one cylinder based on the measurements of the other cylinders and other values such as MAP, RPM, OAT, etc

  model = models.MultiModel([models.KerasModel(), models.XGBModel()])
  #model = models.KerasModel()
  #model = models.XGBModel()


  if not load and not retrain:
    save = False

  if predictOnly:
    load = True
    retrain = False
    save = False

  predictions = {}


  if retrain and save:
    # just save the original dataset elsewhere just in case
    trn.to_csv('model/flights_trn.csv', float_format='%.2f', index=False)
    tst.to_csv('model/flights_tst.csv', float_format='%.2f', index=False)


  global_cht_error = 0
  global_egt_error = 0

  for i in range(1, config.NUMCYLS+1):

    def train(trn, tst, predict_col, remove_col):

      print('scaling...')

      trn_copy = trn.copy()
      tst_copy = tst.copy()

      y_trn = trn_copy[predict_col]
      X_trn = trn_copy.drop([predict_col, remove_col], axis=1)
      y_tst = tst_copy[predict_col]
      X_tst = tst_copy.drop([predict_col, remove_col], axis=1)

      if not predictOnly:
        print('trn info [%s]' % predict_col)
        print(X_trn.describe())
        print(y_trn.describe())
      print('tst info [%s]' % predict_col)
      print(X_tst.describe())
      print(y_tst.describe())


      X_scaler = preprocessing.MinMaxScaler()
      y_scaler = preprocessing.MinMaxScaler()


      X_trn = X_trn.values
      y_trn = y_trn.values.reshape(-1, 1)
      X_tst = X_tst.values
      y_tst = y_tst.values.reshape(-1, 1)


      print('loading/training...')
      if load:
        with open('model/scaler.x.' + predict_col, 'rb') as f: X_scaler = pickle.load(f)
        with open('model/scaler.y.' + predict_col, 'rb') as f: y_scaler = pickle.load(f)
      else:
        X_scaler.fit(X_trn)
        y_scaler.fit(y_trn)

      def scale(X, y):
        X = X_scaler.transform(X)
        y = y_scaler.transform(y)
        X = X.astype('float32')
        y = y.astype('float32')
        return X,y

      X_trn, y_trn = scale(X_trn, y_trn)
      X_tst, y_tst = scale(X_tst, y_tst)

      if load:
        model.Load('model/model.' + predict_col)
      else:
        model.Build()

      if retrain:
        print(plot, plot)
        model.Fit(X_trn, y_trn, X_tst, y_tst, plot)

      if save:
        with open('model/scaler.x.' + predict_col, 'wb') as f: pickle.dump(X_scaler, f)
        with open('model/scaler.y.' + predict_col, 'wb') as f: pickle.dump(y_scaler, f)
        model.Save('model/model.' + predict_col)


      print('evaluating...')
      model.Evaluate(X_tst, y_tst)

      y_prd = model.Predict(X_tst)

      y_prd = y_scaler.inverse_transform(y_prd)
      y_tst = y_scaler.inverse_transform(y_tst)

      mae = mean_absolute_error(y_tst, y_prd)

      predictions[predict_col] = y_prd

      if plot:
        plot_predictions(y_tst, y_prd)

      return mae

    # the goal is to predict CHTs or EGTs from the other cylinders.
    # when predicting CHT for a given cylinder, the corresponding EGT is also removed from the training set
    # and vice versa
    
    cht_col = 'CHT' + str(i)
    egt_col = 'EGT' + str(i)

    e = train(trn, tst, predict_col=cht_col, remove_col=egt_col)
    global_cht_error += e

    e = train(trn, tst, predict_col=egt_col, remove_col=cht_col)
    global_egt_error += e

  global_cht_error /= config.NUMCYLS
  global_egt_error /= config.NUMCYLS

  print('TEST set MAE CHT error = %0.6f' % global_cht_error)
  print('TEST set MAE EGT error = %0.6f' % global_egt_error)


  max_CHT_error = 6
  max_EGT_error = 30

  
  
  if True:
    print('saving...')

    for key, value in predictions.items():
      pred = tst[key+'-PRED'] = value
      diff = tst[key+'-DIFF'] = tst[key+'-PRED'] - tst[key]
      diff_ma = tst[key+'-DIFF-MA6'] = diff.rolling(window=6).mean()

      max_error = max_CHT_error if key.startswith('CHT') else max_EGT_error
      diff_alert = tst[key+'-DIFF-ALERT'] = (diff_ma > max_error).astype(int)

      d = tst[diff_alert>0][['duration', key, key+'-PRED', key+'-DIFF']]
      if(d.shape[0] >= 10): #there is at least 1 minute of unusual values
        print(d.describe())




    tst.to_csv('flights_tstp.csv', float_format='%.2f', index=False)



