import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from gurobipy import Model, GRB
import sys
sys.path.insert(1, 'C:/Users/zwu/OneDrive - IMT Mines Albi/Documents/CodePython/OptimModel')
from functions import *
from gurobipy import Model, GRB

if 'demand_selected' in st.session_state:
    demand_selected = st.session_state['demand_selected']

year = demand_selected.columns[demand_selected.columns.str.startswith("20")][0]

demand_retrofit_center = demand_selected.groupby("Retrofit Center Assigned")[year].sum().to_frame()
# st.write(demand_retrofit_center)

demand_retrofit_center = add_lon_lat(demand_retrofit_center)

st.write("""
# Retrofit Centers and their Demand
""")


fig = px.scatter_mapbox(
    demand_retrofit_center, 
    lat='lat',  # Replace with your actual latitude column name
    lon='lon',  # Replace with your actual longitude column name
    size=year,  # Column to be used for sizing points
    zoom=4,  # Initial zoom level
    center={"lat": 46.2276, "lon": 2.2137},  # Center of the map (France coordinates)
)

st.plotly_chart(fig)
# st.write(demand_retrofit_center)

st.write("""
# Hypotheses and Parameter Setting
- Supposing that the deployment centers can only be located at the chefs-lieux of departments.
- The distance between two places is calculated with Haversine distance (which can be further replaced by road distance using API from Google or OpenStreetMaps).
""")

chef_lieux = st.session_state['chefslieux']
# st.write(chef_lieux)

st.write("""
# Problem Formulation
## Sets and Parameters
- $J$: Set of retrofit centers.
- $I$: Set of potential deployment centers (chefs-lieux of departments).
- $d_{ij}$: Distance between retrofit center $j$ and potential deployment center $i$.
""")

c = st.number_input("Unit transportation cost (euros per kit per km)", value=0.3, step = 0.1, format="%.3f")
C = st.number_input("Opening cost of a deployment center (euros)", value=10000, step=1000)
# - $C_i$: Cost of opening a deployment center at potential location $i$.
st.write("""
- $w_j$: Demand of retrofit center $j$.
- $M$: A large number.

## Decision Variables
- $x_i$: Binary variable indicating whether a deployment center is opened at potential location $i$.
- $f_{ij}$: Quantity of retrofit kits transported from deployment center $i$ to retrofit center $j$.

## Objective Function
- Minimize the cost of transportation and opening deployment centers: 
Minimize $\sum_{i \in I} \sum_{j \in J} d_{ij} f_{ij} + \sum_{i \in I} C * x_i$

## Constraints
- Flow conservation: $$\sum_{i \in I} f_{ij} = w_j$$, for all $$j \in J$$
- Positive flow: $$f_{ij} \geq 0$$, for all $$i \in I$$, for all $$j \in J$$ 
- Deployment Center Opening Constraint: $$f_{ij} \leq M x_i $$, for all $$i \in I$$, for all $$j \in J$$ (where M is a large number)
- Binary constraint: $$x_i \in \{0, 1\}$$, for all $$i \in I$$

# Results
""")

if st.button("Run Optimization"):
    M = 1000000
    model = Model("RetrofitCenterAllocation")

    J = demand_retrofit_center.index.tolist()
    I = chef_lieux.tolist()
    w = demand_retrofit_center[year]

    x = model.addVars(I, vtype=GRB.BINARY, name="x")
    f = model.addVars(I, J, vtype=GRB.CONTINUOUS, name="f")

    model.setObjective(c *sum(distance_calcul(get_coordinates(i), get_coordinates(j)) * f[i,j] for i in I for j in J) + sum(C * x[i] for i in I), GRB.MINIMIZE)

    model.addConstrs(((sum(f[i,j] for i in I) == w[j]) for j in J), name="FlowConservation")
    model.addConstrs((f[i,j] >= 0 for i in I for j in J), name="PositiveFlow")
    model.addConstrs((f[i,j] <= M * x[i] for i in I for j in J), name="DeploymentCenterOpening")

    model.optimize()

    if model.status == GRB.OPTIMAL:
        st.write("Optimal Solution Found!")
        st.write(f"Total Cost: {model.objVal}")
        flow_df = pd.DataFrame([(j, i, f[i, j].x) for i in I for j in J if f[i, j].x > 0], columns=['To', 'From', 'Flow'])
        deployment_center = flow_df.groupby("From")["Flow"].sum().to_frame()
        max_supply = deployment_center['Flow'].max()
        min_marker_size = 1  # Minimum marker size
        max_marker_size = 20  # Maximum marker size
        
        fig = go.Figure()
        max_flow = flow_df['Flow'].max()
        min_line_width = 1  # minimum line width
        max_line_width = 10  # maximum line width
        for index, row in flow_df.iterrows():
            flow = row['Flow']
            line_width = (flow / max_flow) * (max_line_width - min_line_width)
            fig.add_trace(go.Scattermapbox(
                lat=[get_lat(row['From']), get_lat(row['To'])],
                lon=[get_lon(row['From']), get_lon(row['To'])],
                mode='lines',
                line=dict(width=line_width, color='blue'),
                text=f"Flow: {row['Flow']}",
            ))
        for index, row in deployment_center.iterrows():
            supply = row['Flow']
            marker_size = (supply / max_supply) * (max_marker_size - min_marker_size) + min_marker_size

            fig.add_trace(go.Scattermapbox(
                lat=[get_lat(index)],
                lon=[get_lon(index)],
                mode='markers',
                marker=go.scattermapbox.Marker(size=marker_size, color = 'purple'),
                text=f"Supply {supply} units from {get_commune_name(index)}",
                hoverinfo='text'
            ))
        # Set map layout
        fig.update_layout(
            mapbox_style="open-street-map",
            mapbox=dict(
                center={"lat": 46.2276, "lon": 2.2137},  # Adjust based on your data
                zoom=4
            ),
            showlegend=False
        )
        st.plotly_chart(fig)
    



