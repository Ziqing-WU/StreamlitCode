import math
import pandas as pd
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pickle
from gurobipy import Model, GRB, quicksum
from scipy.special import gamma
import matplotlib.pyplot as plt
from itertools import product
from scipy.integrate import odeint


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
Generate a list of communes in Occitanie according to their population
Return a DataFrame with columns: "COM" (Code INSEE Commune), "Commune", "PMUN", "Latitude", "Longitude"
'''

def get_communes_by_population(pop):
    df = pd.read_csv(r"C:\Users\zwu\OneDrive - IMT Mines Albi\Documents\Data\ModelOptim\General\ensemble_population_2021\donnees_communes.csv", encoding='utf-8', sep=';')
    geo = pd.read_csv(r"C:\Users\zwu\OneDrive - IMT Mines Albi\Documents\Data\ModelOptim\General\GeoPosition.csv")
    df_occitanie = df[df['REG']==76]
    df_occitanie = df_occitanie[["COM", "Commune", "PMUN"]]
    df_occitanie = df_occitanie.join(geo.set_index('Code Commune'), on='COM')
    df_occitanie = df_occitanie[df_occitanie['PMUN']>=pop]
    return df_occitanie

'''
Generate a table of contents in the sidebar using the titles of the sections
'''
def create_toc(toc_items=["Hypothesis", "Sets", "Decision Variables", "Parameters", "Objective function", "Constraints", "Datasets", "Results"]):
    st.sidebar.title("Table of Contents")
    for item in toc_items:
        st.sidebar.markdown(f"[{item}](#{item.lower().replace(' ', '-')})")

'''
Calculate the cumulative survival function according to Held et al. 2021
'''

def calculate_cum_survival_function(a, beta_shape=6, gamma_scale=15.2):
    gamma_term = gamma(1+1/beta_shape)
    csf = np.exp((-(a/gamma_scale)**beta_shape) * gamma_term**beta_shape)
    return csf

# The differential equation described by the Bass model
def bass_diff(N, t, p, q, M):
    dNdt = (p + q * N / M) * (M - N)
    return dNdt
