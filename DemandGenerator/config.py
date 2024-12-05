import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import pyproj
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os
from datetime import date, datetime
import csv
import sys
from email.mime import base
import matplotlib as mpl
import graphviz
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import re
import pickle
from scipy.integrate import odeint
import gurobipy as gp
import math
from gurobipy import Model, GRB, quicksum
from itertools import product
import folium
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster
from collections import defaultdict


current_year = 2024
# The existing product data source
source_path = r'C:\Users\zwu\Documents\Data\Vehicle\occitanie_cleaned_code_INSEE.csv'
# The folder containing all the intermediate data for each step
precharged_folder = r'C:\Users\zwu\Documents\StreamlitCode\DemandGenerator\data_precharged'
executive_factor_folder = r'C:\Users\zwu\Documents\Data\ExecutiveFactors'

def change_type(df_vehicles, to_numeric=['poids_a_vide_national','cylindree','niv_sonore','co2', 'puissance_net_maxi', 'Age']):
    df_vehicles['date_premiere_immatriculation'] = pd.to_datetime(df_vehicles['date_premiere_immatriculation'], errors='coerce', format='%d/%m/%Y').dt.normalize()
    for col in to_numeric:
        df_vehicles[col] = pd.to_numeric(df_vehicles[col], errors='coerce')
    return df_vehicles

def create_map():
    # This file comes from the geoservices product Admin Express COG 2022
    fp = rf'{executive_factor_folder}\Ref-1-CodeGeo\COMMUNE_OCCITANIE.shp'
    map_df = gpd.read_file(fp)
    # map_df.to_crs(pyproj.CRS.from_epsg(4326), inplace=True)
    return map_df[['INSEE_COM', 'geometry']].set_index('INSEE_COM')


def get_communes_by_population(pop):
    df = pd.read_csv(rf"{executive_factor_folder}\donnees_communes_population.csv", encoding='utf-8', sep=';',
        dtype={'COM': str})
    geo = pd.read_csv(rf"{executive_factor_folder}\Ref-1-CodeGeo\GeoPosition.csv")
    df_occitanie = df[df['REG']==76]
    df_occitanie = df_occitanie[["COM", "Commune", "PMUN"]]
    df_occitanie = df_occitanie.join(geo.set_index('Code Commune'), on='COM')
    df_occitanie = df_occitanie[df_occitanie['PMUN']>=pop]
    return df_occitanie

'''
Calculate the distance between two points on the Earth's surface with longitude and latitude in degrees
'''
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = R * c
    return d

'''
Calculate the cumulative distribution function of Weibull with parameters proposed by Held et al. 2021
'''
def calculate_cum_distribution_function(tau, beta_shape=6, gamma_mean=15.2):
    lamda = gamma_mean / math.gamma(1+1/beta_shape)
    return 1- np.exp(-(tau/lamda)**beta_shape)