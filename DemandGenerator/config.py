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
from datetime import date
import csv
import sys

current_year = 2024
# The existing product data source
source_path = r'C:\Users\zwu\Documents\Data\Vehicle\occitanie_cleaned_code_INSEE.csv'
# The folder containing all the intermediate data for each step
precharged_folder = r'C:\Users\zwu\Documents\Data\Vehicle\data_precharged'

def change_type(df_vehicles, to_numeric=['poids_a_vide_national','cylindree','niv_sonore','co2', 'puissance_net_maxi', 'Age']):
    df_vehicles['date_premiere_immatriculation'] = pd.to_datetime(df_vehicles['date_premiere_immatriculation'], errors='coerce', format='%d/%m/%Y').dt.normalize()
    for col in to_numeric:
        df_vehicles[col] = pd.to_numeric(df_vehicles[col], errors='coerce')
    return df_vehicles