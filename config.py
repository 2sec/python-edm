#!/usr/bin/python
# -*- coding: utf-8 -*-

random_state = 5719

NUMCYLS = 6


# there was a problem with the oil pressure sensor during those flights, repaired afterwards
def invalidOilP(fnum, date, values):
    invalidOilP = 1 if fnum in [56, 57, 58] else 0
    if invalidOilP: values['OILP'] = -1
    return invalidOilP


additional_columns = { 'INVALID_OILP': invalidOilP }
