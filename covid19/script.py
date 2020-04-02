import os
import csv
import operator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
sns.set()


################
# SIMPLE PARSING
################

files = ['confirmed', 'deaths', 'recovered']

for s in files:
    temp = pd.read_csv('./COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_{}_global.csv'.format(s))
    temp = temp.drop('Lat', axis=1)
    temp = temp.drop('Long', axis=1)
    temp = temp.sort_values(by=['Country/Region'])
    temp = temp[~temp['Province/State'].str.contains(',', na=False)]
    temp = temp.groupby(['Country/Region']).sum()
    temp = temp.transpose()
    temp['date'] = pd.date_range(start='22/1/2020', periods=len(temp), freq='D')
    if s == 'Confirmed':
            confirmed = temp
    elif s == 'Deaths':
            deaths = temp
    elif s == 'Recovered':
            recovered = temp
    temp.to_csv("{}.tsv".format(s), sep="\t")



