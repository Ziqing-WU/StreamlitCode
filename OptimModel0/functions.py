from geopy.distance import great_circle
import pandas as pd
import streamlit as st

pop_den = pd.read_csv("pop_den.csv")
pop_den.set_index('Code Commune', inplace=True)

def parse_coordinates(cd):
    lon = cd.strip('[]').split(',')[0]
    lat = cd.strip('[]').split(',')[1]
    return lon, lat

# Calculate the distance between two coordinates
def distance_calcul(cd_1, cd_2):
    lon_1, lat_1 = parse_coordinates(cd_1)
    lon_2, lat_2 = parse_coordinates(cd_2)
    location1 = (lat_1, lon_1) 
    location2 = (lat_2, lon_2)
    distance = great_circle(location1, location2).kilometers
    return distance

# Add the latitude and longitude to the dataframe indexed by code commune INSEE. 
def add_lon_lat(dataframe):
    dataframe = dataframe.join(pop_den[["coordinates"]], how='left')
    dataframe['lon'] = dataframe['coordinates'].str.strip('[]').str.split(',').str[0].astype(float)
    dataframe['lat'] = dataframe['coordinates'].str.strip('[]').str.split(',').str[1].astype(float)
    dataframe.drop(columns = ["coordinates"], inplace=True)
    return dataframe

# With a commune code, retrieve the coordinates of the mairie
def get_coordinates(code_commune):
    return pop_den.loc[code_commune]['coordinates']

def get_lon(code_commune):
    return pop_den.loc[code_commune]['coordinates'].strip('[]').split(',')[0]

def get_lat(code_commune):
    return pop_den.loc[code_commune]['coordinates'].strip('[]').split(',')[1]

# With a commune code, retrieve the name of the commune
def get_commune_name(code_commune):
    return pop_den.loc[code_commune].iloc[1]