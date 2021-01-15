# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 11:34:42 2020

@author: jaely
"""

#Load necessary libraries
import numpy as np
import pandas as pd
import os
from shapely.geometry import Point
import matplotlib.pyplot as plt
import geopandas
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from netCDF4 import Dataset 

#Reset wd
os.chdir('C:\\Users\\jaely\\Documents\\Research_2020')

#Read in DRTO locations
drto_locs = pd.read_csv('DRTO_locations.csv')

#Make coordinates plottable
drto_locs['coordinates'] = drto_locs[['longitude', 'latitude']].values.tolist()
drto_locs['coordinates'] = drto_locs['coordinates'].apply(Point)
drto_locs = geopandas.GeoDataFrame(drto_locs, geometry='coordinates')

#Read in water temp data (cleaned up in R)
drto_temps = pd.read_csv('drto_num.csv')
print(drto_temps.shape)

#Drop useless columns
drto_temps = drto_temps.drop(drto_temps.columns[[0,1]],axis=1)
print(drto_temps.shape)

#Read in times
drto_times = pd.read_csv('drto_time.csv',header=None,names=["t"],parse_dates=True)
dstrip = lambda x: datetime.strptime(x,'%m/%d/%Y %H:%M')
drto_times['t2'] = drto_times['t'].apply(dstrip)

#Read in geomorphic and benthic data
os.chdir('C:\\Users\\jaely\\Documents\\Research_2020\\ACA_DRTO_data')
geomorphic = geopandas.read_file('geomorphic.geojson')
benthic = geopandas.read_file('benthic.geojson')
#Plot
#fig, ax = plt.subplots(1, figsize=(20,20))
#base = geomorphic.plot(ax=ax, color='lightblue')
#drto_locs.plot(ax=base,color="red")
#_ = ax.axis('off')

#fig, ax = plt.subplots(1, figsize=(20,20))
#base = benthic.plot(ax=ax, color='lightblue')
#drto_locs.plot(ax=base,color="red")
#_ = ax.axis('off')

#Convert CRS for loc points to match geomorphic data
drto_locs.crs = {'init': 'epsg:4326'}

#Extract geo values to location points
joined = geopandas.sjoin(drto_locs, geomorphic, how='inner', op='intersects')    
print(joined['class'].value_counts())


#Export joined table to csv
joined.to_csv("drto_joined.csv",index=False,mode='w')

#Rename columns because "class" is a reserved word
joined.columns= ['Point', 'latitude', 'longitude', 'elevation', 'coordinates','index_right', 'reefclass']

#Make list of site names from joined table
sites = joined['Point']
splt = lambda x: x.split("_")
sites = sites.apply(splt)
keep = lambda x: x[1]
sites=sites.apply(keep)
drto_locs['Point'] = sites
joined['Point'] = sites

#Split up point names in the temps dataset
pointnames = pd.Series(drto_temps.columns)
pointnames = pointnames.apply(splt)
keep0 = lambda x: x[0]
pointnames = pointnames.apply(keep0)

#Throw out temps points outside the study area
keep = pd.Series([pointnames[i] in sites.values for i in np.arange(0,len(pointnames),1)])
temps_filt = drto_temps.drop(drto_temps.columns[~keep],axis=1)
pointnames = pd.Series(temps_filt.columns)
pointnames = pointnames.apply(splt)
keep0 = lambda x: x[0]
pointnames = pointnames.apply(keep0)

#Sort "joined" by alphabetical order of points
joined = joined.sort_values(by = 'Point')

#Aggregate the filtered temps dataset in a sane way
test_keys = list(temps_filt.columns)
test_values = list (pointnames)

#Make a dictionary with a mapping
site_dic = {test_keys[i]: test_values[i] for i in range(len(test_keys))} 

#Group columns from same site by mean
temps_filt =temps_filt.groupby(site_dic, axis=1).mean()
pointnames = pd.Series(temps_filt.columns)

#Create rcs: a list of reef classes equal to the number
#of points in the filtered temps dataset
rc = lambda x:joined.reefclass[pointnames[x]==sites].iloc[0]
rcs = pd.Series([rc(i) for i in np.arange(0,len(pointnames),1)])

#Plot all sites over time by color
#Make color variable
classcors = {"Reef Slope":"limegreen", "Reef Crest":"gold","Back Reef Slope":"c","Sheltered Reef Slope":"dodgerblue","Outer Reef Flat":"orangered","Plateau":"orange"}
cors = rcs.replace(classcors)

legend_stuff = [Line2D([0], [0], marker='o', color='w', label='Slope', markerfacecolor='limegreen', markersize=8),
                Line2D([0], [0], marker='o', color='w', label='Crest', markerfacecolor='gold', markersize=8),
                Line2D([0], [0], marker='o', color='w', label='Back Slope', markerfacecolor='c', markersize=8),
                Line2D([0], [0], marker='o', color='w', label='Sheltered Slope', markerfacecolor='dodgerblue', markersize=8),
                Line2D([0], [0], marker='o', color='w', label='Outer Flat', markerfacecolor='orangered', markersize=8),
                Line2D([0], [0], marker='o', color='w', label='Plateau', markerfacecolor='orange', markersize=8),]

for i in np.arange(0,len(cors),1):
    plt.scatter(drto_times['t2'],temps_filt.iloc[:,i],c=cors[i],marker='.',s=0.1)
    plt.xlabel("Time")
    plt.ylabel("Temp degrees C")
    plt.title("Temperature over time")
    plt.legend(handles = legend_stuff,loc='lower left')

#Get first and last time points for each site
times = drto_times['t2']

temps_filt = temps_filt[times < '2013-12-31 00:00:00']
times = times[times < '2013-12-31 00:00:00']
temps_filt = temps_filt[times > '2008-06-01 00:00:00']
times =  times[times > '2008-06-01 00:00:00']

#Daily max and min with boxplots
days = times.dt.date
daysu = times.dt.date.unique()

dmin = lambda x: temps_filt[days==x].min(axis=0)
dailymin = pd.DataFrame([dmin(daysu[i]) for i in np.arange(0,len(daysu),1)])
boxes = sns.boxplot(data=dailymin,palette=cors)
boxes.set(xlabel= "site", ylabel = "Degrees C")
boxes.set_title("Daily minimum temperature by site")

dmax = lambda x: temps_filt[days==x].max(axis=0)
dailymax = pd.DataFrame([dmax(daysu[i]) for i in np.arange(0,len(daysu),1)])
boxes = sns.boxplot(data=dailymax,palette=cors)
boxes.set(xlabel= "site", ylabel = "Degrees C")
boxes.set_title("Daily maximum temperature by site")

#Daily flux with boxplots
dailydiff = dailymax - dailymin 
boxes = sns.boxplot(data=dailydiff,palette=cors)
boxes.set(xlabel= "site", ylabel = "Degrees C")
boxes.set_title("Daily differences in temperature by site")
    
#Annual 99th percentile with boxplots
years = times.dt.year
yearsu = times.dt.year.unique()

y99 = lambda x: temps_filt[years==x].quantile(q=0.99,axis=0)
year99 = pd.DataFrame([y99(yearsu[i]) for i in np.arange(0,len(yearsu),1)])

boxes = sns.boxplot(data=year99,palette=cors)
boxes.set(xlabel= "site", ylabel = "Degrees C")
boxes.set_title("99th percentile annual temps by site")

#Load MMM data from NOAA
ncin = Dataset('ct5km_climatology_mmm.nc', 'r', format='NETCDF4')
print(ncin.dimensions.keys())
print(ncin.variables.keys())

mmm_nc = ncin.variables['sst_clim_mmm'][:]
print(mmm_nc.long_name)
print(mmm_nc.units)
clim_lats = ncin.variables['lat'][:]
clim_lons = ncin.variables['lon'][:]
ncin.close()
