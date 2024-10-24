from xml.etree.ElementInclude import default_loader
import streamlit as st
import numpy as np
import pandas as pd
import geopandas as gpd
import pyproj
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

"""
# Total Addressable Market
"""

current_year = 2024
file_path = r"C:\Users\zwu\Documents\Data\Vehicle\data_precharged\vehicle_list_TM.csv"
csv_file = st.file_uploader(
    'Upload a file of the existing product information representing the total market'
    )

@st.cache_data
def load_vehicle_data(path):
    df = pd.read_csv(path, low_memory=False, dtype=object)
    return df

if not csv_file:
    df = load_vehicle_data(file_path)
else:
    df = pd.read_csv(csv_file, low_memory=False, dtype=object)

st.write(df.head(),df["categorie_vehicule"].unique())

vehicle_engine_type =  df["energie"].unique().tolist()
vehicle_engine_type_default = ["ES","GO"]
vehicle_engine_type_dict = {
    "Abbreviation": ["ES", "EL", "GO", "EG", "FE", "EN", "EE", "GG", "FL", "GN", "EH", "GH", "GP", "GL", "AC", "GE", "FG", "GA", "PE", "FN", "HE", "FH", "GQ", "EQ"],
    "Description": [
        "Gasoline", "Electric", "Diesel", "Dual-fuel Gasoline-LPG", "Super Ethanol", 
        "Dual-fuel Gasoline-Natural Gas", "Hybrid Electric (rechargeable)", "Gasogene-Diesel Mix", 
        "Super Ethanol-Electric (rechargeable)", "Natural Gas", "Hybrid Electric (non-rechargeable)", 
        "Hybrid Diesel-Electric (non-rechargeable)", "Liquefied Petroleum Gas (LPG)", 
        "Hybrid Diesel-Electric (rechargeable)", "Air Compressed", "Gasogene-Gasoline Mix", 
        "Dual-fuel Super Ethanol-LPG", "Gasogene", "Monofuel LPG-Electric (rechargeable)", 
        "Dual-fuel Super Ethanol-Natural Gas", "Hybrid Electric (rechargeable)", 
        "Hybrid Super Ethanol (non-rechargeable)", "Diesel-Natural Gas Mix (dual fuel) and Electric (non-rechargeable)", 
         "Dual-fuel Gasoline-LPG and Electric (non-rechargeable)"
    ]
}
category = df["categorie_vehicule"].unique().tolist()
category_default = ['M1', 'N1']
locations = ['31555', '45234', '45273', '75056', '81004']
car_bodies = ['AA', 'AB', 'AC', 'AD', 'AE', 'AF', 'AG', 'BA', 'BB', 'BC', 'BD', 'BE', 'BX', 'CA', 'CB', 'CD', 'CE', 'CF', 'CG', 'CH', 'CI', 'CJ', 'CX', 'DA', 'DB', 'DC', 'DE']
norm_euro = ['Euro ' + str(i) for i in range(1,8)]
colors = ['Black', 'Silver', 'Gray', 'Blue']


st.write("Filter 1: Legislative and Technical Constraints")
with st.expander("Legislative Constraints"):
    category_leg = st.multiselect('Vehicle Category', category, category)
    st.write("For the explanation on vehicle types, refer to: https://www.legifrance.gouv.fr/codes/section_lc/LEGITEXT000006074228/LEGISCTA000006129091/2016-04-15/")
    min_age_leg = st.slider('Vehicle Minimum Age',
                    min_value=0,
                    max_value=20,
                    value=5
                    )
    engine_type_tech = st.multiselect('Type of Fuel or Energy Source', vehicle_engine_type, default = vehicle_engine_type_default,key=1)
    geo_leg = st.multiselect('Geographical Coverage', locations, locations)

with st.expander("Technical Constraints"):
    engine_type_tech = st.multiselect('Type of Fuel or Energy Source', vehicle_engine_type, default = vehicle_engine_type_default,key=2)
    category_tech = st.multiselect('Vehicle Category', category, default=['M1', 'N1'])
    min_age_tech, max_age_tech = st.select_slider('Vehicle Age Range',
                                                    options=np.array([i for i in range(1,21)]),
                                                    value=[5,15])
    min_ep_tech, max_ep_tech = st.select_slider(
                    'Range of Maximum Net Power',
                    options = np.array([i for i in range(0, 301)]),
                    value = [60, 110]
                    )
    min_MPLW, max_MPLW = st.select_slider('Range of Maximum Permissible Laden Weight (MPLW) (kg)', 
                                            options=np.array([i*100 for i in range(10, 201)]),
                                            value = [1300, 4400])
    min_GVWR, max_GVWR = st.select_slider('Range of Gross Vehicle Weight Rating (GVWR) (kg)',
                                        options=np.array([i*100 for i in range(10, 201)]),
                                        value= [1300, 5300])
    min_pv, max_pv = st.select_slider('Range of Mass of Vehicle in Running Order (kg)',
                                        options=np.array([i*100 for i in range(10, 201)]),
                                        value=[1000, 4000])
    car_body_type = st.multiselect('Car Body Style CE', car_bodies,  ['AA', 'AB', 'AC', 'AD', 'AE', 'AF', 'AG'])
    st.markdown("Check [this website](https://www.cartegrise.com/carte-grise-detail/carrosserie) for more information")
    min_engine_size, max_engine_size = st.select_slider('Range of Vehicle Engine Size (cm²)',
                                                        options=np.array([i*100 for i in range(10, 1610)]),
                                                        value=[1000, 14000])
    min_lev_noise, max_lev_noise = st.select_slider('Range of Noise Level at Standstill (dB)', 
                                                    options=np.array([i for i in range(30, 121)]),
                                                    value=[45, 60])
    min_CO2, max_CO2 =  st.select_slider('Range of CO2 emission in Cruising Conditions (g/km)', 
                                                    options=np.array([i for i in range(20, 201)]),
                                                    value=[80, 200])
    ind_class_env = st.multiselect('European Emission Standards', norm_euro, norm_euro)
    color = st.multiselect('Color', colors, colors)
    st.markdown("**Gearbox type information is not directly available from Vehicle Registration System.** It can be found via le Code National d'Identification de Véhicule (CNIT) (the fourth character). For a specific vehicle, this information can be found via [this website](http://www.type-mine.com/).")
        
def intersection(list1, list2):
    return list(set(list1) & set(list2))

engine_type = intersection(engine_type_leg, engine_type_tech)
category = intersection(category_leg, category_tech)
min_age = min(min_age_leg, min_age_tech)
max_age = max_age_tech
geo = geo_leg
min_ep = min_ep_tech
max_ep = max_ep_tech      
        
st.markdown(
    """
    # Total Addressable Market
    From total market to total addressable market, legislative and technical constraints are considered. 
    The parameters to be set are shown on the top of this page. 
    They serve as filters for **all visualizations** on this page.
    """
)


@st.cache_data
def apply_filters(df):
    # Age
    df = df[(df['Age']>min_age)]
    # Engine Type
    df = df[df['Ft'].isin(engine_type)]
    # Geo
    df = df.astype({'Code_commune': 'str'})
    df = df[df['Code_commune'].isin(geo)]
    # Category
    #TODO
    # Engine Power
    df = df[(df['ep (KW)']<max_ep) & (df['ep (KW)']>min_ep)]
    return df
    
df = apply_filters(df)

st.metric(
    'Number of Vehicles in the Total Addressable Market',
    df.count()[0]
)

st.markdown(
    '''
    ## Geographical Distribution & Model Selection
    '''
)

@st.cache
def construct_map(file_path):
    map_df = gpd.read_file(file_path)
    map_df.to_crs(pyproj.CRS.from_epsg(4326), inplace=True)
    return map_df[['INSEE_COM', 'geometry']].set_index('INSEE_COM')

fp = r'C:\Users\zwu\OneDrive - IMT Mines Albi\Documents\Data\DemandForecasting\Ref-1-CodeGeo\ADMIN-EXPRESS-COG_3-1__SHP__FRA_WM_2022-04-15\ADMIN-EXPRESS-COG\1_DONNEES_LIVRAISON_2022-04-15\ADECOG_3-1_SHP_WGS84G_FRA\COMMUNE.shp'
map_df = construct_map(fp)
group_commune_df = df.groupby('Code_commune').count().iloc[:, :1].rename(columns={'Country':'Num Vehicles'})

df_merged = map_df.merge(group_commune_df, left_index=True, right_index=True).sort_values(by='Num Vehicles', ascending=False)    
fig_map = px.choropleth_mapbox(df_merged, geojson = df_merged['geometry'], color='Num Vehicles', locations=df_merged.index, mapbox_style="carto-positron", zoom=4, center = {"lat": 47, "lon": 2}, color_continuous_scale='Greys')

@st.cache
def group_by(df):
    df = df.groupby(by=['Mh', 'Cn']).count().iloc[:,:1].rename(columns={'Country':'Num Vehicles'})
    df.reset_index(inplace=True)
    df.sort_values(by='Num Vehicles', ascending=False, inplace=True)
    return df

group_df = group_by(df)
subfig = make_subplots(specs=[[{"secondary_y": True}]])
group_model_df = group_df.groupby(by='Cn').sum().sort_values(by = 'Num Vehicles', ascending=False)
group_model_df['Cumulative Percentage'] = group_model_df.cumsum()/group_model_df.sum()*100

fig = px.bar(group_model_df, y='Num Vehicles')


pareto_line = px.scatter(group_model_df, y='Cumulative Percentage')
pareto_line.update_traces(yaxis="y2")

col1, col2 = st.columns([1,1])

cumperc = st.number_input('How many percent of vehicle fleet that we want to cover?', value=80, step=5, max_value=100, min_value=0, format='%i')

subfig.add_traces(fig.data + pareto_line.data)
subfig.layout.yaxis.title="Num Vehicles"
subfig.layout.yaxis2.title="Cumulative Percentage"
subfig.add_shape(type='line',
                x0=0,
                y0=cumperc,
                x1=400,
                y1=cumperc,
                line=dict(color='Gray',),
                xref='x',
                yref='y2')


with col2:
    st.plotly_chart(fig_map)
with col1:
    st.plotly_chart(
        subfig
    )

model_considered = group_model_df[group_model_df['Cumulative Percentage']<cumperc].index

group_model_df_filtered = group_df[group_df.Cn.isin(model_considered)]

nb_model = group_model_df[group_model_df['Cumulative Percentage']<cumperc].count().iloc[0]
st.write(
    'The number of models covered is ',
    nb_model
)

@st.cache
def convert_df(df):
    return df.to_csv().encode('utf-8')

csv = convert_df(group_model_df_filtered)

col1, col2 = st.columns([1,1])

with col1:
    st.write("Here is the list of ", nb_model, " models")
    group_model_df_filtered
    st.download_button("Download Model List Here", csv, mime='text/csv', file_name=str(current_year)+'_model_list_TAM.csv')
with col2:
    df
    csv_df = convert_df(df)
    st.download_button("Download Vehicle List Here", csv_df, mime='text/csv', file_name=str(current_year)+'_vehicle_list_TAM.csv')
