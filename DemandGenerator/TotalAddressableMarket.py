from config import *

with st.sidebar:
    """
    Navigation

    [Total Addressable Market](#total-addressable-market)
    - [Apply Filter 1: Legislative and Technical Constraints](#apply-filter-1-legislative-and-technical-constraints)
    - [Geographical Distribution](#geographical-distribution)
    - [Model Selection](#model-selection)
    """

"""
# Total Addressable Market

From total market to total addressable market, legislative and technical constraints are considered. 
The parameters are to be set for legislative and technical constraints. 
They serve as filters for **all visualizations** on this page.
"""

file_path = precharged_folder + r"\vehicle_list_TM.csv"
csv_file = st.file_uploader(
    'Upload a file of the existing product information representing the total market'
    )
"A dataset on ICE vehicles, filtered with default settings to represent the total market, has been loaded by default if no data is provided."

@st.cache_data
def load_vehicle_data(path):
    df = pd.read_csv(path, low_memory=False, index_col=0, dtype=object)
    return df

if not csv_file:
    df = load_vehicle_data(file_path)
    st.write("Here is a preview of the imported data:", df.head(5))
else:
    df = pd.read_csv(csv_file, low_memory=False, dtype=object)

vehicle_engine_type =  df["energie"].unique().tolist()
vehicle_engine_type_default_tech = ["ES","GO"]
vehicle_engine_type_default_leg = [
    "ES", "EG", "EN", "FE", "EE", "EH", "FG", "FL", "FN", 
    "EQ", "GE", "GO", "GL", "GH", "GG"
]
category = df["categorie_vehicule"].unique().tolist()
category_default = ['M1', 'N1']
locations = df["code_commune_titulaire"].unique().tolist()
pays_titulaire = df["pays_titulaire"].unique().tolist()
pays_titulaire_default = ["FRANCE"]
car_bodies = df["carrosserie_ce"].unique().tolist()

st.write("## Apply Filter 1: Legislative and Technical Constraints")
with st.expander("Legislative Constraints"):
    category_leg = st.multiselect('Vehicle Category', category, category, key=3)
    st.write("For the explanation on vehicle types, please refer to: https://www.legifrance.gouv.fr/codes/section_lc/LEGITEXT000006074228/LEGISCTA000006129091/2016-04-15/")
    min_age_leg = st.slider('Vehicle Minimum Age',
                    min_value=0,
                    max_value=20,
                    value=5
                    )
    engine_type_leg = st.multiselect('Type of Fuel or Energy Source', vehicle_engine_type, default = vehicle_engine_type_default_leg,key=1)
    st.write("For the explanation on type of fuel, please refer to: https://www.portail-cartegrise.fr/h/champ-p3-carte-grise")
    geo = st.multiselect('Registration Country', pays_titulaire, "FRANCE")
    st.write("The default value for this field is set to FRANCE if left blank.")

with st.expander("Technical Constraints"):
    engine_type_tech = st.multiselect('Type of Fuel or Energy Source', vehicle_engine_type, default = vehicle_engine_type_default_tech,key=2)
    st.write("For the explanation on type of fuel, please refer to: https://www.portail-cartegrise.fr/h/champ-p3-carte-grise")

    category_tech = st.multiselect('Vehicle Category', category, default=category_default)
    min_age_tech, max_age_tech = st.select_slider('Vehicle Age Range',
                                                    options=np.array([i for i in range(1,21)]),
                                                    value=[5,15])
    min_ep, max_ep = st.select_slider(
                    'Range of Maximum Net Power',
                    options = np.array([i for i in range(0, 301)]),
                    value = [60, 110]
                    )
    min_pv, max_pv = st.select_slider('Range of Mass of Vehicle in Running Order (kg)',
                                        options=np.array([i*100 for i in range(10, 201)]),
                                        value=[1000, 4000])
    car_body_type = st.multiselect('Car Body Style CE', car_bodies,  ['AA', 'AB', 'AC', 'AD', 'AE', 'AF'])
    st.markdown("Check [this website](https://www.cartegrise.com/carte-grise-detail/carrosserie) for more information")
    min_engine_size, max_engine_size = st.select_slider('Range of Vehicle Engine Size (cm²)',
                                                        options=np.array([i*100 for i in range(10, 1610)]),
                                                        value=[1000, 14000])
    min_lev_noise, max_lev_noise = st.select_slider('Range of Noise Level at Standstill (dB[A])', 
                                                    options=np.array([i for i in range(30, 121)]),
                                                    value=[45, 85])
    min_CO2, max_CO2 =  st.select_slider('Range of CO2 emission in Cruising Conditions (g/km)', 
                                                    options=np.array([i for i in range(20, 250)]),
                                                    value=[80, 200])
    # st.markdown("**Gearbox type information is not directly available from Vehicle Registration System.** It can be found via le Code National d'Identification de Véhicule (CNIT) (the fourth character). For a specific vehicle, this information can be found via [this website](http://www.type-mine.com/).")
        
def intersection(list1, list2):
    return list(set(list1) & set(list2))

# Convert df data type
df = change_type(df)

engine_type = intersection(engine_type_leg, engine_type_tech)
category = intersection(category_leg, category_tech)
min_age = max(min_age_leg, min_age_tech)
max_age = max_age_tech

@st.cache_data
def apply_filters(df):
    filter_steps = []
    filter_steps.append(("Total","", df.shape[0]))
    # Age
    df = df[(df['Age']>=min_age) & (df['Age']<=max_age)] 
    filter_steps.append(("Age","Total", df.shape[0]))
    # Engine Type
    df = df[df['energie'].isin(engine_type)]
    filter_steps.append(("Engine Type","Age", df.shape[0]))
    # Geo
    df = df[(df['pays_titulaire'].isin(geo)) | (df['pays_titulaire'].isna())]
    filter_steps.append(("Registration Country","Engine Type", df.shape[0]))
    # Category
    df = df[df['categorie_vehicule'].isin(category)]
    filter_steps.append(("Vehicle Category","Registration Country", df.shape[0]))
    # Engine Power
    df = df[(df['puissance_net_maxi']<=max_ep) & (df['puissance_net_maxi']>=min_ep)]
    filter_steps.append(("Engine Power", "Vehicle Category", df.shape[0]))
    # Poids a vide national
    df = df[(df['poids_a_vide_national']<=max_pv) & (df['poids_a_vide_national']>=min_pv)]
    filter_steps.append(("Weight","Engine Power", df.shape[0]))
    # Carroserie CE
    df = df[df['carrosserie_ce'].isin(car_body_type)]
    filter_steps.append(("Car Body Type","Weight", df.shape[0]))
    # Cylindree
    df = df[(df['cylindree']>=min_engine_size) & (df['cylindree']<=max_engine_size)]
    filter_steps.append(("Engine Size","Car Body Type", df.shape[0]))
    # Niveau sonore
    df = df[(df['niv_sonore']>=min_lev_noise) & (df['niv_sonore']<=max_lev_noise)]
    filter_steps.append(("Noise Level","Engine Size", df.shape[0]))
    # CO2
    df = df[(df['co2']>=min_CO2) & (df['co2']<=max_CO2)]
    filter_steps.append(("CO2 Emissions","Noise Level", df.shape[0]))
    return df, filter_steps
    
df, filter_steps = apply_filters(df)
filter_df = pd.DataFrame(filter_steps, columns=["Filter Step", "Previous Step", "Remaining Count"])
fig = px.funnel(filter_df, y='Filter Step', x='Remaining Count')
fig.update_layout(xaxis_title='Number of filtered vehicles')
st.plotly_chart(fig)


st.metric(
    'Number of Vehicles in the Total Addressable Market',
    df.shape[0]
)

st.markdown(
    '''
    ## Geographical Distribution
    '''
)

@st.cache_data
def create_map():
    # This file comes from the geoservices product Admin Express COG 2022
    fp = r'C:\Users\zwu\Documents\Data\CommuneShape\COMMUNE_OCCITANIE.shp'
    map_df = gpd.read_file(fp)
    # map_df.to_crs(pyproj.CRS.from_epsg(4326), inplace=True)
    return map_df[['INSEE_COM', 'geometry']].set_index('INSEE_COM')

map_df = create_map()
group_commune_df = df.groupby('code_commune_titulaire').count().iloc[:, :1]
group_commune_df.columns.values[0]="Num Vehicles"

df_merged = map_df.merge(group_commune_df, left_index=True, right_index=True).sort_values(by='Num Vehicles', ascending=False)    
fig_map = px.choropleth_mapbox(df_merged, geojson = df_merged['geometry'], color='Num Vehicles', locations=df_merged.index, mapbox_style="carto-positron", zoom=5, center = {"lat": 45.5, "lon": 2}, color_continuous_scale='Greys')
fig_map.update_traces(marker_line_width=0)

st.plotly_chart(fig_map)
@st.cache_data
def group_by(df):
    df = df.groupby(by=['type_version_variante']).count().iloc[:,:1]
    df.columns.values[0] = "Num Vehicles"
    df.sort_values(by='Num Vehicles', ascending=False, inplace=True)
    return df

group_df = group_by(df)
subfig = make_subplots(specs=[[{"secondary_y": True}]])
group_df['Cumulative Percentage'] = group_df.cumsum()/group_df.sum()*100

fig = px.bar(group_df, y='Num Vehicles')

pareto_line = px.scatter(group_df, y='Cumulative Percentage')
pareto_line.update_traces(yaxis="y2")
st.write("## Model Selection")
cumperc = st.number_input('How many percent of vehicle fleet that we want to cover?', value=50, step=5, max_value=100, min_value=0, format='%i')

subfig.add_traces(fig.data + pareto_line.data)
subfig.layout.yaxis.title="Num Vehicles"
subfig.layout.yaxis2.title="Cumulative Percentage"
subfig.add_shape(type='line',
                x0=0,
                y0=cumperc,
                x1=20000,
                y1=cumperc,
                line=dict(color='Gray',),
                xref='x',
                yref='y2')
subfig.layout.height = 800
st.plotly_chart(subfig)

with st.expander("Show only the graph of cumulative percentage of vehicles vs. number of TVVs"):
    df_pareto = group_df.sort_values(by="Num Vehicles", ascending=False).reset_index(drop=True)
    df_pareto['Num TVVs'] = df_pareto.index + 1

    fig = go.Figure()
    fig = px.scatter(
        df_pareto,
        x='Num TVVs',
        y='Cumulative Percentage',
        title='Cumulative Percentage of Vehicles vs. Number of TVVs'
    )

    st.plotly_chart(fig)

model_considered = group_df[group_df['Cumulative Percentage']<cumperc].index
group_model_df_filtered = group_df[group_df.index.isin(model_considered)]

nb_model = group_df[group_df['Cumulative Percentage']<cumperc].count().iloc[0]
st.write(
    'The number of models covered is ',
    nb_model
)

@st.cache_data
def convert_df(df):
    return df.to_csv().encode('utf-8')

csv = convert_df(group_model_df_filtered)

col1, col2 = st.columns([1,1])

with col1:
    st.write("Here is the list of ", nb_model, " models")
    group_model_df_filtered
    st.download_button("Download Model List Here", csv, mime='text/csv', file_name=str(current_year)+'_model_list_TAM.csv')
with col2:
    st.write(df.head(10))
    csv_df = convert_df(df)
    st.download_button("Download Vehicle List Here", csv_df, mime='text/csv', file_name=str(current_year)+'_vehicle_list_TAM.csv')
    """
    *N.B.* In this step, the considered filters in vehicle dataset are only legislative and technical ones. 
    The model selection section provides a tool to generate a model list for which the company aims to provide retrofit service. 
    This list will be used in the serviceable addressable market step.
    """