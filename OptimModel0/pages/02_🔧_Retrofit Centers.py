import streamlit as st
import pandas as pd
import numpy.random as rd
import numpy as np
import geopandas as gpd
import pyproj
import plotly.express as px
<<<<<<< HEAD:OptimModel0/pages/02_🔧_Retrofit Centers.py
from gurobipy import Model, GRB
import sys
sys.path.insert(1, 'C:/Users/zwu/OneDrive - IMT Mines Albi/Documents/CodePython/OptimModel')
from functions import *

demand_data_types = {
    'Code commune de résidence': str,
}
demand = pd.read_csv("parc_auto_pop_den.csv", dtype=demand_data_types)[["Code commune de résidence", "Commune de résidence", "lon", "lat", "Carburant", "Crit\'Air", "num_retrofit", "2023", "2024", "2025", "2026", "2027"]]

columns = st.columns([1,1,1])
with columns[0]:
    year = st.slider("Which year?", 2023, 2027, step=1)
with columns[1]:
    critair = st.multiselect("Which Crit'Air?", ['Crit\'Air 3', 'Crit\'Air 4', 'Crit\'Air 5', 'Non classé'], ['Non classé'])
with columns[2]:
    carburant = st.multiselect("Which source of energy?", ["Diesel", "Essence"], ["Essence"])

demand_selected = demand[["Code commune de résidence", "Commune de résidence", "lon", "lat", "Carburant", "Crit\'Air", "num_retrofit",str(year)]]
demand_selected = demand_selected[demand_selected["Carburant"].isin(carburant)]
demand_selected = demand_selected[demand_selected["Crit\'Air"].isin(critair)]
demand_selected = demand_selected[demand_selected[str(year)]!=0]

st.dataframe(demand_selected[['Code commune de résidence', 'Commune de résidence', 'Carburant', 'Crit\'Air', 'num_retrofit', str(year)]])

fig = px.scatter_mapbox(
    demand_selected, 
    lat='lat',  # Replace with your actual latitude column name
    lon='lon',  # Replace with your actual longitude column name
    color='Crit\'Air',  # Column to be used for color coding
    size=str(year),  # Column to be used for sizing points
    hover_name='Commune de résidence',  # Column to show on hover
    hover_data=['Carburant'],
    zoom=4,  # Initial zoom level
    center={"lat": 46.2276, "lon": 2.2137},  # Center of the map (France coordinates)
)

st.plotly_chart(fig)

demand_selected_groupby = demand_selected[["Code commune de résidence", str(year)]].groupby("Code commune de résidence").sum()
demand_selected_groupby = demand_selected_groupby[demand_selected_groupby[str(year)]!=0]

st.write(f'Total demand to be covered is {demand_selected_groupby[str(year)].sum()}. They are from {demand_selected_groupby.shape[0]} locations.')

st.write("""
         # Hypotheses and Parameter Setting
         - Supposing that the retrofit centers can only be located at the chefs-lieux of departments. 
         - The distance between two places is calculated with Haversine distance (which can be further replaced by road distance using API from Google or OpenStreetMaps). 
         """)

columns = st.columns([1,1])
with columns[0]:
    radius = st.number_input("Coverage radius (km)", min_value = 20, max_value = 80, step = 10, value = 50)
with columns[1]:
    max_num = st.number_input("Maximal number of facilities", value = 10, step = 5)

# st.write("## Generate Distance Matrix")
file_path = r"C:\Users\zwu\OneDrive - IMT Mines Albi\Documents\Data\DemandForecasting\Ref-1-CodeGeo\departement_2022.csv"
chefslieux = pd.read_csv(file_path)["CHEFLIEU"]
st.session_state['chefslieux'] = chefslieux
demand_selected_groupby.reset_index(inplace=True)
distance_matrix = pd.merge(demand_selected_groupby["Code commune de résidence"], chefslieux, how='cross')
columns_rename = {"Code commune de résidence": "Demand Point", "CHEFLIEU": "Retrofit Center"}
distance_matrix.rename(columns=columns_rename, inplace=True)
pop_den = pd.read_csv("pop_den.csv")

# st.dataframe(pop_den)

distance_matrix = pd.merge(distance_matrix, pop_den[["Code Commune", "coordinates"]], how = "left", left_on="Demand Point", right_on="Code Commune")
distance_matrix = pd.merge(distance_matrix, pop_den[["Code Commune", "coordinates"]], how = "left", left_on="Retrofit Center", right_on="Code Commune")
distance_matrix = distance_matrix[["Demand Point", "Retrofit Center", "coordinates_x", "coordinates_y"]]
columns_rename = {"coordinates_x": "coordinates_demand", "coordinates_y": "coordinates_retrofit"}
distance_matrix.rename(columns=columns_rename, inplace=True)

distance_matrix['Distance_km'] = distance_matrix.apply(lambda row : distance_calcul(row['coordinates_demand'], row['coordinates_retrofit']), axis=1)
# st.dataframe(distance_matrix)

st.write("""
        # Problem Formulation
        ## Sets and Parameters
        - $J$: Set of demand points (communes).
        - $I$: Set of potential retrofit center locations (chefs-lieux of departments).
        - $d_{ij}$​: Distance between demand point $j$ and facility location $i$.
        - $R$: Coverage radius (maximum distance at which a demand point is considered covered by a retrofit center).
        - $Q$: Number of facilities to be established.
        - $w_j$​: Demand at demand point $j$.
        ## Decision Variables
        - $x_i$​: Binary variable that equals 1 if a facility is established at location $i$, and 0 otherwise.
        - $y_j$: Binary variable for demand point $j$ being covered.
        
        ## Objective Function
        Maximize the total covered demand:
        Maximize $$\sum_{j \in J} w_j y_j $$
         
        ## Constraints
        - Demand coverage: $y_j \leq \sum_{i \in \{I : d_{ij}\leq R\}} x_i $ for all $j\in J$
        - Number of retrofit centers limited: $\sum_{i \in I} x_i \leq Q$
        - Binary constraints: $x_i \in \{0, 1\}$ for all $i \in I$, $y_j \in \{0, 1\}$ for all $j \in J$
         
        # Results

         """) 

if st.button("Run the optimization model!"):
    # Initialize the model
    m = Model("MCLP")

    J = distance_matrix["Demand Point"].unique()
    I = distance_matrix["Retrofit Center"].unique()

    # Decision variables
    x = m.addVars(I, vtype=GRB.BINARY, name="x")
    y = m.addVars(J, vtype=GRB.BINARY, name="y")

    # Objective function: Maximize total covered demand
    m.setObjective(sum(demand_selected_groupby[demand_selected_groupby["Code commune de résidence"]==j][str(year)].iloc[0] * y[j] for j in J), GRB.MAXIMIZE)

    # Constraints
    # Demand coverage
    for j in J:
        m.addConstr(y[j] <= sum(x[i] for i in I if distance_matrix[(distance_matrix["Demand Point"] == j) & (distance_matrix["Retrofit Center"] == i)]["Distance_km"].iloc[0] <= radius))

    # Number of retrofit centers limited
    m.addConstr(sum(x[i] for i in I) <= max_num)

    # Solve the model
    m.optimize()
    st.write("Optimization Runtime: {:.2f} seconds".format(m.Runtime))

    # Extract the solution
    solution = {i: x[i].X for i in I if x[i].X>0.5}
    demand_points = {j: y[j].X for j in J if y[j].X>0.5}
    st.session_state['solution'] = solution
    st.session_state['demand_points'] = demand_points

    if m.status == GRB.Status.OPTIMAL:
        # Output the objective function value
        demand_covered = m.ObjVal
        st.write(f"Demand covered {demand_covered}")
    else:
        st.write("Optimal solution was not found.")

    # st.write(solution)
    # st.write()

    solution = pd.DataFrame(list(solution.items()), columns=['INSEE_Code', 'x'])

    pop_den.set_index('Code Commune', inplace=True)
    solution = solution.set_index('INSEE_Code')
    solution = add_lon_lat(solution)
    
    demand_selected["covered_status"] = demand_selected["Code commune de résidence"].isin(demand_points.keys())
    st.session_state['demand_selected'] = demand_selected
=======
import json
import requests
>>>>>>> parent of ff9b015 (uncovered demand shown):OptimModel/pages/02_🔧_Retrofit Centers.py



<<<<<<< HEAD:OptimModel0/pages/02_🔧_Retrofit Centers.py
    fig.add_scattermapbox(
        lat=solution['lat'],
        lon=solution['lon'],
        mode='markers',
        marker=dict(symbol='grocery'),
        name='Retrofit Center'
    )
    fig.update_layout(showlegend=False)

    st.plotly_chart(fig)

    # st.write(demand_selected)

def find_retrofit_center(code_commune, covered_status, list_retrofit_center):
    if covered_status:
        id_distance_matrix = distance_matrix[distance_matrix["Demand Point"]==code_commune][distance_matrix["Retrofit Center"].isin(list_retrofit_center)][distance_matrix["Distance_km"]<=radius]["Distance_km"].idxmin()
        return distance_matrix.loc[id_distance_matrix]["Retrofit Center"]
    else:
        return None
    
if st.button("Use this result for determining deployment centers!"):
    if 'solution' in st.session_state and 'demand_points' in st.session_state:
        solution = st.session_state['solution']
        list_retrofit_center = list(solution.keys())
        demand_selected = st.session_state['demand_selected']
        demand_selected["Retrofit Center Assigned"] = demand_selected.apply(lambda row: find_retrofit_center(row["Code commune de résidence"], row["covered_status"], list_retrofit_center), axis=1)
        st.session_state['demand_selected'] = demand_selected
    else:
        st.error("Please run the optimization model first.")
=======
Code_INSEE = '81004'
api_url = f'https://geo.api.gouv.fr/communes/{Code_INSEE}?fields=code&format=geojson&geometry=mairie'
response = requests.get(api_url)
if response.status_code == 200:
    st.write(response.json()["geometry"]["coordinates"])
>>>>>>> parent of ff9b015 (uncovered demand shown):OptimModel/pages/02_🔧_Retrofit Centers.py
