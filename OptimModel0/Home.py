import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go

st.title("Problem Settings")
st.image("Network.png")
token = "pk.eyJ1IjoiemlxaW5nd3UiLCJhIjoiY2xvaGUwZXhpMDlycDJqcWoxbWxzbzN0YiJ9.dAj7sXo6Is9HRrue9VO8WA"
px.set_mapbox_access_token(token)

maturity_level = ['Complying', 'Integrating', 'Establishing']
features = ['Recover products, materials and energy as locally possible', 'Explore new circular functionalities of existing facilities', 'Circulate materials and products with hyperconnected logistics system', 'Materialize objects on-demand in PI open production fabs', 'Deploy open and hyperconnected sustainability performance monitoring']
features_with_linebreaks = ['Recover products, materials <br> and energy as locally possible', 'Explore new circular <br> functionalities of <br> existing facilities', 'Circulate materials <br> and products with <br> hyperconnected <br> logistics system', 'Materialize objects <br> on-demand in PI <br> open production fabs', 'Deploy open <br> and hyperconnected <br> sustainability <br> performance <br> monitoring']
st.write("Select maturity levels for the following HCSC features:")
col1, col2 = st.columns([1,1])
maturity_level_value = [0,0,0,0,0]
with col1:
    for i in range(3):
        maturity_level_value[i] = st.select_slider(features[i], key=i, options=maturity_level)

with col2:
    for i in [3, 4]:
        maturity_level_value[i] = st.select_slider(features[i], key=i, options=maturity_level)

def maturity_to_num(maturity):
    if maturity == "Complying":
        return 0
    if maturity == "Integrating":
        return 1
    if maturity == "Establishing":
        return 2

maturity_level_value = [maturity_to_num(maturity_level_value[i]) for i in range(len(maturity_level_value))]

col1, col2 = st.columns([1,1])
with col1:
    df = pd.DataFrame(dict(
    r=maturity_level_value,
    theta=features_with_linebreaks))
    fig = px.line_polar(df, r='r', theta='theta', line_close=True)
    fig.update_layout(
        margin=dict(l=70, r=70, t=50, b=50),
        polar=dict(
            radialaxis=dict(
                tickvals=[0, 1, 2],
                ticktext=['Complying', 'Integrating', 'Establishing'],
                range=[0, 2]
            ),
            angularaxis=dict(
                tickfont=dict(
                    size=10  # Adjust the font size for theta labels
                )
        )
        ))
    st.plotly_chart(fig, use_container_width=True)

# Helper function to generate random latitude and longitude
def generate_location(country='France'):
    if country == 'France':
        # Approximate bounding box for mainland France
        lat = np.random.uniform(43.1, 49.5)
        long = np.random.uniform(-1, 7)
    elif country == 'China':
        # Approximate bounding box for China
        lat = np.random.uniform(18.0, 54.0)
        long = np.random.uniform(73.0, 135.0)
    elif country == 'Sweden':
        # Approximate bounding box for Sweden
        lat = np.random.uniform(55.0, 69.0)
        long = np.random.uniform(11.0, 24.0)
    return lat, long

# Create the dataframe with fake data
facilities = ['battery supplier', 'powertrain supplier', 'mechanical supplier', 'factory',
              'distribution center', 'retrofit center', 'recycling center', 'market']

# Count of each facility
facility_counts = {'battery supplier': 2, 'powertrain supplier': 2, 'mechanical supplier': 4,
                   'factory': 3, 'distribution center': 10, 'retrofit center': 30, 
                   'recycling center': 20, 'market': 50}

# Generate the data
data = {'type_facility': [], 'lat': [], 'long': [], 'fixed_cost': [], 'unit_cost': [],
        'fixed_emission': [], 'unit_emission': [], 'if_battery': []}

for facility, count in facility_counts.items():
    for _ in range(count):
        country = 'China' if facility == 'battery supplier' and len(data['type_facility']) == 0 else 'Sweden' if facility == 'battery supplier' and len(data['type_facility']) == 1 else 'France'
        data['type_facility'].append(facility)
        lat, long = generate_location(country)
        data['lat'].append(lat)
        data['long'].append(long)
        data['fixed_cost'].append(np.random.uniform(5000, 20000))
        data['unit_cost'].append(np.random.uniform(50, 150))
        data['fixed_emission'].append(np.random.uniform(200, 1000))
        data['unit_emission'].append(np.random.uniform(5, 20))
        data['if_battery'].append(facility in ['battery supplier', 'distribution center', 'factory', 'retrofit center'])

supply_chain_df = pd.DataFrame(data)



# Marker settings based on the type of facility
marker_symbols = {
    'battery supplier': 'industry',
    'powertrain supplier': 'industry',
    'mechanical supplier': 'industry',
    'factory': 'industry',
    'distribution center': 'warehouse',
    'retrofit center': 'car_repair',
    'recycling center': 'recycling',
    'market': 'grocery'
    # Add other facility types and their corresponding symbols here
}
supply_chain_df['marker_symbol'] = supply_chain_df['type_facility'].map(marker_symbols)

# Display the DataFrame
st.write(supply_chain_df)  

with col2:
    fig = px.scatter_mapbox(
        supply_chain_df,
        lat='lat',
        lon='long',
        hover_name='type_facility',
        hover_data=['fixed_cost', 'unit_cost', 'fixed_emission', 'unit_emission'],
        color='type_facility',  # Optional: to have different colors for different facilities
        color_discrete_sequence=px.colors.qualitative.Set1,  # Optional color sequence
        mapbox_style="carto-positron",
        zoom=4,  # Adjust zoom to show desired area, e.g., France
        center={"lat": 46.2276, "lon": 2.2137}  # Center on France
    )
    # fig.update_traces(marker=dict(symbol = 'bus'))
    st.plotly_chart(fig, use_container_width=True)
