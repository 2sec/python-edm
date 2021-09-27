#!/usr/bin/python
# -*- coding: utf-8 -*-

random_state = 5719

NUMCYLS = 6

max_CHT_error = 5
max_EGT_error = 20


# there was a problem with the oil pressure sensor during those flights, repaired afterwards
def invalidOilP(fnum, date, values):
    invalidOilP = 1 if fnum in [56, 57, 58] else 0
    if invalidOilP: values['OILP'] = -1
    return invalidOilP


# replaced flexible baffle seals on 15/7
# todo: seal rigid baffles correctly
def bafflesVersion(fnum, date, values):
    #if fnum >= XX: return 2
    if fnum >= 70: return 1
    return 0
    


additional_columns = { 'INVALID_OILP': invalidOilP, 'BAFFLES_VERSION' : bafflesVersion}


