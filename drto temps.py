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
import geopandas as gpd
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from netCDF4 import Dataset 
import numpy.ma as npm

#Reset wd
os.chdir('C:\\Users\\jaely\\Documents\\Research_2020')

#Read in DRTO locations
drto_locs = pd.read_csv('DRTO_locations.csv')

#Make coordinates plottable
drto_locs['coordinates'] = drto_locs[['longitude', 'latitude']].values.tolist()
drto_locs['coordinates'] = drto_locs['coordinates'].apply(Point)
drto_locs = gpd.GeoDataFrame(drto_locs, geometry='coordinates')

#Sort in alphabetical order and reindex
drto_locs = drto_locs.sort_values(by = 'Point')
drto_locs = drto_locs.reset_index(drop = True)

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
geomorphic = gpd.read_file('geomorphic.geojson')

#Convert CRS for loc points to match geomorphic data
drto_locs.crs = {'init': 'epsg:4326'}

#Extract geo values to location points
joined = gpd.sjoin(drto_locs, geomorphic, how='inner', op='intersects')    
print(joined['class'].value_counts())
joined = joined.reset_index(drop = True)

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
joined['Point'] = sites

#Split up point names in the temps dataset
pointnames = pd.Series(drto_temps.columns)
pointnames = pointnames.apply(splt)
keep0 = lambda x: x[0]
pointnames = pointnames.apply(keep0)

#Throw out temps points outside the study area
in_geo = pd.Series([pointnames[i] in sites.values for i in np.arange(0,len(pointnames),1)])
temps_filt = drto_temps.drop(drto_temps.columns[~in_geo],axis=1)
pointnames = pd.Series(temps_filt.columns)
pointnames = pointnames.apply(splt)
keep0 = lambda x: x[0]
pointnames = pointnames.apply(keep0)

#Aggregate the filtered temps dataset by point
#Points with two simultanouesly active temperature
#loggers are aggregated by mean
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
classcors = {"Reef Slope":"limegreen", "Reef Crest":"gold","Back Reef Slope":"c","Sheltered Reef Slope":"mediumblue","Outer Reef Flat":"orangered","Plateau":"orange"}
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

#Save time variable under new name
times = drto_times['t2']

#Pull out times
temps_filt = temps_filt[times < '2013-12-31 00:00:00']
times = times[times < '2013-12-31 00:00:00']
temps_filt = temps_filt[times > '2008-06-01 00:00:00']
times =  times[times > '2008-06-01 00:00:00']

#Remove columns with significant NAs in timeframe
temps_filt = temps_filt.drop(temps_filt.columns[[18,19,20]],axis=1)
joined_filt = joined.drop(joined.index[[18,19,20]],axis=0)

#Delete unnecessary variables to avoid confusion
del pointnames,rcs, cors

#Remake rcs and color variables of correct length
pointnames2 = temps_filt.columns
rc = lambda x:joined.reefclass[pointnames2[x]==sites].iloc[0]
rcs2 = pd.Series([rc(i) for i in np.arange(0,len(pointnames2),1)])
cors2 = rcs2.replace(classcors)

#Daily max and min with boxplots
days = times.dt.date
daysu = times.dt.date.unique()

dmin = lambda x: temps_filt[days==x].min(axis=0)
dailymin = pd.DataFrame([dmin(daysu[i]) for i in np.arange(0,len(daysu),1)])
boxes = sns.boxplot(data=dailymin,palette=cors2)
boxes.set(xlabel= "site", ylabel = "Degrees C")
boxes.set_title("Daily minimum temperature by site")

dmax = lambda x: temps_filt[days==x].max(axis=0)
dailymax = pd.DataFrame([dmax(daysu[i]) for i in np.arange(0,len(daysu),1)])
boxes = sns.boxplot(data=dailymax,palette=cors2)
boxes.set(xlabel= "site", ylabel = "Degrees C")
boxes.set_title("Daily maximum temperature by site")

#Daily flux with boxplots
dailydiff = dailymax - dailymin 
boxes = sns.boxplot(data=dailydiff,palette=cors2)
boxes.set(xlabel= "site", ylabel = "Degrees C")
boxes.set_title("Daily differences in temperature by site")
    
#Annual 99th percentile with boxplots
years = times.dt.year
yearsu = times.dt.year.unique()

y99 = lambda x: temps_filt[years==x].quantile(q=0.99,axis=0)
year99 = pd.DataFrame([y99(yearsu[i]) for i in np.arange(0,len(yearsu),1)])

boxes = sns.boxplot(data=year99,palette=cors2)
boxes.set(xlabel= "site", ylabel = "Degrees C")
boxes.set_title("99th percentile annual temps by site")

#Load MMM data from NOAA
os.chdir('C:\\Users\\jaely\\Documents\\Research_2020')
joined = joined.reset_index()

ncin = Dataset('ct5km_climatology_mmm.nc', 'r', format='NETCDF4')
print(ncin.dimensions.keys())
print(ncin.variables.keys())

mmm_nc = ncin.variables['sst_clim_mmm']

print(mmm_nc.long_name)
print(mmm_nc.units)
mmm_nc = mmm_nc[0,:,:]

clim_lats = ncin.variables['lat'][:]
clim_lons = ncin.variables['lon'][:]
ncin.close()

#Function to find closest lat or lon value in mmm set to location

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

lats = [find_nearest(clim_lats,joined_filt['latitude'][i]) for i in np.arange(0,len(joined_filt['latitude']),1)]
lons = [find_nearest(clim_lons,joined_filt['longitude'][i]) for i in np.arange(0,len(joined_filt['longitude']),1)]

#Pull out MMMs for each point
mmms = mmm_nc[lats,lons].data

#Calculate daily mean temperatures
dmean = lambda x: temps_filt[days==x].mean(axis=0)
dailymeans = pd.DataFrame([dmean(daysu[i]) for i in np.arange(0,len(daysu),1)])

#Calculate degree heating weeks per day
dhw = lambda x,y: ((round(dailymeans.iloc[x,y]-mmms[y])/7) if ((dailymeans.iloc[x,y]-mmms[y])>=1) else 0)
dhw_data = [[dhw(i,j) for i in np.arange(0,len(dailymeans.index),1)] for j in np.arange(0,len(mmms),1)]
dhw_data = pd.DataFrame(dhw_data).transpose()
dhw_data.columns = dailymeans.columns

#Summarize into DHWs per year
dys = times.groupby(days).min().dt.year

yDHW = lambda x: dhw_data[dys.values==x].sum(axis=0)
yearlyDHW = pd.DataFrame([yDHW(yearsu[i]) for i in np.arange(0,len(yearsu),1)])

boxes = sns.boxplot(data=yearlyDHW,palette=cors2)
boxes.set(xlabel= "site", ylabel = "Degrees C")
boxes.set_title("Annual DHWs temps by site")

#There are some serious problems with the interpretation of this that I will address
#at a later date