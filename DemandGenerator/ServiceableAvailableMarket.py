from config import *


with st.sidebar:
    """
    Navigation

    [Serviceable Available Market](#serviceable-available-market)
    - [Apply Filter 2: Strategies and Business Model of the Company](#apply-filter-2-strategies-and-business-model-of-the-company)
    """
st.markdown(
    '''
    # Serviceable Available Market
    '''
)

csv_vehicles = st.file_uploader(
    'Upload a file of the existing product information representing the total addressable market'
)
"A dataset on ICE vehicles, filtered with default settings to represent the total addressable market, has been loaded by default if no data is provided."

if csv_vehicles is not None:
    df_vehicles = pd.read_csv(csv_vehicles, low_memory=False, index_col=0, dtype=object)
else:
    file_path_v= precharged_folder + f"\{str(current_year)}_vehicle_list_TAM.csv"
    df_vehicles = pd.read_csv(file_path_v, low_memory=False, index_col=0, dtype=object)

df_vehicles = change_type(df_vehicles)
st.write("Here is a preview of the imported data:", df_vehicles.head(5))

locations = df_vehicles["code_commune_titulaire"].unique().tolist()

categorys = df_vehicles["categorie_vehicule"].unique().tolist()

st.write("""

## Apply Filter 2: Strategies and Business Model of the Company
**Vehicle Model**
""")

csv_file = st.file_uploader(
    'Upload a list of models or select all of them'
    )
if csv_file is not None:
    df_model = pd.read_csv(csv_file, low_memory=False, index_col=0, dtype=object)
else:
    file_path = precharged_folder + f"\{str(current_year)}_model_list_TAM.csv"
    df_model = pd.read_csv(file_path, low_memory=False, index_col=0, dtype=object)

select_all = st.checkbox("Select all the models")

"**Geographical Coverage**"

geo = st.multiselect('INSEE commune codes', locations, [])
if st.toggle("Upload a list of INSEE commune codes", value=True):
    csv_codes_INSEE = st.file_uploader("Upload a csv file containing INSEE commune codes")
    if csv_codes_INSEE is not None:
        geo = pd.read_csv(csv_codes_INSEE, dtype=object)
    else:
        file_path_g = precharged_folder + f"\geo_couv_BM.csv"
        geo = pd.read_csv(file_path_g, dtype=object)
st.write("By default, communes with population densities ranging from petites villes to grands centres urbains are selected, representing less than 10% of the commune codes but more than 60% of the population.")

category = st.multiselect('Vehicle Category', categorys, 'M1')

df_model = df_model.index

@st.cache_data
def apply_filters(df):
    filter_steps = []
    filter_steps.append(("Total","", df.shape[0]))
    # filter geographical coverage
    if not select_all:
        df = df[df['code_commune_titulaire'].isin(geo["COM"])]
    filter_steps.append(("Geo","Total", df.shape[0]))
    # filter models
    df = df[df['type_version_variante'].isin(df_model)]
    filter_steps.append(("Vehicle Model","Geo", df.shape[0]))
    # filter categories
    df = df[df['categorie_vehicule'].isin(category)]
    filter_steps.append(("Category","Vehicle Model", df.shape[0]))
    return df,filter_steps

df_vehicles, filter_steps = apply_filters(df_vehicles)
filter_df = pd.DataFrame(filter_steps, columns=["Filter Step", "Previous Step", "Remaining Count"])
fig = px.funnel(filter_df, y='Filter Step', x='Remaining Count')
fig.update_layout(xaxis_title='Number of filtered vehicles')
st.plotly_chart(fig)

@st.cache_data
def convert_df(df):
    return df.to_csv().encode('utf-8')

csv = convert_df(df_vehicles)

st.download_button('Download the list of vehicles in Service Available Market here', csv, mime='text/csv', file_name=str(current_year)+'_vehicle_list_SAM.csv')




