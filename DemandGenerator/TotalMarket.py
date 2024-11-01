from config import *

with st.sidebar:
    """
    Navigation

    [Total Market](#total-market)
    - [Select Columns of Interests from the Dataset](#select-columns-of-interests-from-the-dataset)
    - [Geographical Distribution](#geographical-distribution)
    - [Vehicle Age & Brand](#vehicle-age-brand)
    """


@st.cache_data
def create_map():
    # This file comes from the geoservices product Admin Express COG 2022
    fp = rf'{executive_factor_folder}\Ref-1-CodeGeo\COMMUNE_OCCITANIE.shp'
    map_df = gpd.read_file(fp)
    # map_df.to_crs(pyproj.CRS.from_epsg(4326), inplace=True)
    return map_df[['INSEE_COM', 'geometry']].set_index('INSEE_COM')

@st.cache_data
def load_vehicle_data():
    fp_vehicle = r'C:\Users\zwu\Documents\Data\Vehicle\occitanie_cleaned_code_INSEE.csv'
    df = pd.read_csv(fp_vehicle, low_memory=False, dtype=object)
    return df

st.markdown(
    """
    # Total Market
    """
)

csv_file = st.file_uploader(
    'Upload a file of the existing product information'
    )

st.markdown(
    """
    To showcase the use of the demand estimation framework, a dataset on ICE vehicles has been loaded by default if no data is provided.
    """
)

if not csv_file:
    df = load_vehicle_data()
else:
    df = pd.read_csv(csv_file, low_memory=False, dtype=object)

"""
## Select Columns of Interests from the Dataset
"""

options = st.multiselect("Select the columns that you think relevant for estimating the demand", df.columns[2:], ["code_commune_titulaire","pays_titulaire", "marque", "date_premiere_immatriculation", "type_version_variante", "poids_a_vide_national", "categorie_vehicule", "carrosserie_ce", "cylindree", "puissance_net_maxi", "energie", "niv_sonore", "co2", "classe_env"])

@st.cache_data
def select_vehicle_data(df, options):
    df_selected = df[options]
    df_selected = change_type(df_selected, ['poids_a_vide_national','cylindree','niv_sonore','co2', 'puissance_net_maxi'])
    return df_selected

df_selected = select_vehicle_data(df, options)
st.write("Understanding the data:")
dtypes = df_selected.dtypes

object_cols = dtypes[dtypes == 'object'].index
numeric_cols = dtypes[dtypes == 'float64'].index

datetime_cols = dtypes[dtypes == 'datetime64[ns]'].index

if st.button("Visualize categorical data"):
    for col in object_cols:
        value_counts = df_selected[col].value_counts()
        if len(value_counts) > 20:
            value_counts = value_counts.nlargest(20)
        plt.figure(figsize=(10,5))
        sns.barplot(x=value_counts.index, y=value_counts.values)
        plt.xticks(rotation=45)
        plt.title(f"Distribution of {col}")
        st.pyplot(plt)

if st.button("Visualize numerical data"):
    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(10,5))
        sns.boxplot(df_selected[col], ax=ax)
        ax.set_title(f"Distribution of {col}")
        ax.tick_params(axis='x', rotation=45) 
        st.pyplot(plt)

if st.button("Visualize datetime data"):
    for col in datetime_cols:
        st.subheader(f'Distribution of {col}')
        # Extract year to make it easier to visualize
        df_selected['year'] = df_selected[col].dt.year
        plt.figure(figsize=(10,5))
        num_bins = int(df_selected['year'].max() - df_selected['year'].min())
        sns.histplot(df_selected['year'], bins=num_bins, kde=False)
        plt.title(f"Distribution of {col} by Year")
        st.pyplot(plt)

if st.button("Check the null values in the data"):
    st.write(df_selected.isna().sum())

st.markdown(
    """
    ## Geographical Distribution
    """
)

map_df = create_map()
group_commune_df = df_selected.groupby('code_commune_titulaire').count().iloc[:, :1]
group_commune_df.columns.values[0]="Num Vehicles"
group_brand_df = df_selected.groupby(['marque', 'type_version_variante', 'code_commune_titulaire']).count().iloc[:, :1]
group_brand_df.columns.values[0]="Num Vehicles"
group_brand_df.sort_values(by='Num Vehicles', ascending=False,inplace=True)
group_brand_df = group_brand_df.reset_index(level=['marque','type_version_variante', 'code_commune_titulaire'])

df_merged = map_df.merge(group_commune_df, left_index=True, right_index=True).sort_values(by='Num Vehicles', ascending=False)    

n = st.number_input("The number of cities to be displayed by default",min_value=1, value="min", step=5)
geo_filters = st.multiselect(
    'Select cities to be analysed',
    df_merged.index,
    np.array(df_merged.index[:n])
)

fig = px.bar(group_brand_df[group_brand_df['code_commune_titulaire'].isin(geo_filters)], x='type_version_variante', y='Num Vehicles', hover_data=['code_commune_titulaire', 'Num Vehicles','marque'])
fig.update_xaxes(range=(-.5,15))
df_merged = df_merged[df_merged.index.isin(geo_filters)]
fig_map = px.choropleth_mapbox(df_merged, geojson = df_merged['geometry'], color='Num Vehicles', locations=df_merged.index, mapbox_style="carto-positron", zoom=5.5, center = {"lat": 44, "lon": 2}, color_continuous_scale='Greys')
fig_map.update_traces(marker_line_width=0)
col1, col2 = st.columns([1, 1.5])

with col1:
    st.plotly_chart(fig)
with col2:    
    st.plotly_chart(fig_map)


st.markdown(
    """
    ## Vehicle Age & Brand
    """
)
df_selected['Age']= current_year - df_selected['date_premiere_immatriculation'].dt.year
group_age_df = df_selected.groupby(['Age', 'marque', 'type_version_variante']).count().iloc[:, :1]
group_age_df.columns.values[0] = 'Num Vehicles'
group_age_df.reset_index(['marque', 'type_version_variante'], inplace=True)
group_age_df.sort_values('Num Vehicles', inplace=True, ascending=False)

col1, col2, col3 = st.columns([2,2,1])

with col1:
    brand_filters = st.multiselect(
        'Select brands to be analysed',
        group_age_df['marque'].unique(),
        ['RENAULT', 'VOLKSWAGEN']
    )
    if st.checkbox("Select all the brands"):
        brand_filters = None
    

with col2:
    min_age, max_age = st.select_slider(
                        'Select vehicle age range',
                        options = np.array([i for i in range(1, 25)]),
                        value = [1, 15]
                        )   
if brand_filters:
    group_age_df = group_age_df[group_age_df['marque'].isin(brand_filters)]
group_age_df = group_age_df[(group_age_df.index>min_age) & (group_age_df.index<max_age)]
# if brand_filters != None:
#     group_commune_df_filtered = df_selected[df_selected['marque'].isin(brand_filters)]
# else:
#     group_commune_df_filtered = df_selected
# group_commune_df_filtered = group_commune_df_filtered[(df_selected.Age>min_age) & (df_selected.Age<max_age)]
# group_commune_df_filtered = group_commune_df_filtered.groupby(by = 'code_commune_titulaire').count().iloc[:,:1]
# group_commune_df_filtered.columns.values[0] = 'Num Vehicles'
# df_merged = map_df.merge(group_commune_df_filtered, left_index=True, right_index=True).sort_values(by='Num Vehicles', ascending=False)    

with col3:
    st.metric('Number of Vehicles Selected', group_age_df['Num Vehicles'].sum())

# col1, col2 = st.columns([1,1])

# with col1:
st.plotly_chart(
    px.bar(group_age_df, y='Num Vehicles', hover_data=['marque', 'type_version_variante'])
)

# with col2:
#     st.plotly_chart(
#         px.choropleth_mapbox(df_merged, geojson = df_merged['geometry'], color='Num Vehicles', locations=df_merged.index, mapbox_style="carto-positron", zoom=5.5, center = {"lat": 43.5, "lon": 2}, color_continuous_scale='Greys') 
#     )

st.markdown(
    """
    A preview for the dataset Total Market produced:
    """
)

st.write(df_selected.head(10))

@st.cache_data
def convert_df(df):
    return df.to_csv().encode('utf-8')

csv = convert_df(df_selected)

st.download_button("Download", csv, mime='text/csv', file_name='vehicle_list_TM.csv')
st.write("*N.B.* In this step, the relevant columns of the dataset of existing products have been selected. The previous two sections Geographical Distribution and Vehicle Age & Brand are only for data understanding which won't change the output of this page.")
st.write("After downloading the data for the Total Market. Let's continue with Total Addressable Market (TAM).")
if st.button("Go to TAM"):
    st.switch_page("TotalAddressableMarket.py")

