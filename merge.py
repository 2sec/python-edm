#!/usr/bin/python
# -*- coding: utf-8 -*-

import config

import pandas as pd
import os
import sys


from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    argc = len(sys.argv)
    if(argc > 1):
        csvdir = sys.argv[1]
        if not csvdir.endswith('/'): csvdir += '/'
        files = os.listdir(csvdir)
        files = [f for f in files if f.endswith('.csv')]

        flights = None

        for f in files:
            print('reading', f)
            data = pd.read_csv(csvdir + f, delimiter = ',')

            data['date'] = pd.to_datetime(data['date'])
            data['month'] = data['date'].dt.month

            date = data['date'].iloc[0]
            data['date'] -= date

            cols = []
            for i in range(0, config.NUMCYLS):
                cols.append('EGT' + str(i+1))
                cols.append('CHT' + str(i+1))

            data['duration'] = (data['date'].dt.total_seconds() // 60).astype('int32')


            FlightDuration = data['duration'].iloc[-1]
            rows = data.shape[0]

            if FlightDuration >= 30:
                # ignore first 7 and last 3 minutes of the flight
                data = data[data['duration'] >= 7]
                data = data[data['duration'] <= FlightDuration - 3]
                
                cols = ['duration', 'month', 'OILP', 'OILT', 'OAT', 'FF', 'MAP', 'RPM', 'CRB', 'HP'] + cols
                data = data[cols]

                print('Flight Date', date, '- Duration', FlightDuration, 'min', '- Rows', rows)

                if flights is None: 
                    flights = data
                else:
                    flights = pd.concat([flights, data])

        flights.to_csv('flights.csv', float_format='%.2f', index=False)

        print(flights.info())
        print(flights.describe())


        flights_trn, flights_tst = train_test_split(flights, test_size=0.1, random_state=config.random_state)        

        flights_trn.to_csv('flights_trn.csv', float_format='%.2f', index=False)
        flights_tst.to_csv('flights_tst.csv', float_format='%.2f', index=False)



