import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    layout='wide',
    initial_sidebar_state='auto',
    page_icon='🌓'
)

locations = ['31555', '45234', '45273', '75056', '81004']
categorys = ['L', 'M1', 'N1']
consumers = ['toB', 'toC']

st.write("Filter 2: Strategies and Business Model of the Company")

current_year = st.radio('Show Market Situation in Year ', [2022, 2023, 2024, 2025, 2026], horizontal=True)
csv_file = st.file_uploader(
    'Upload a list of models or select all'
    )
select_all = st.checkbox("Select all the models")
geo = st.multiselect('Geographical Coverage', locations, ['31555', '45234', '75056', '81004'])
category = st.multiselect('Vehicle Category', categorys, 'M1')
consumer = st.multiselect('Consumer Type', consumers, consumers)

    
if csv_file is not None:
    df_model = pd.read_csv(csv_file)
else:
    file_path = r"C:\Users\zwu\OneDrive - IMT Mines Albi\Documents\Data\DemandForecasting\Vehicles\Simulated_car_registration\\"+str(current_year)+"_model_list_TAM.csv"
    df_model = pd.read_csv(file_path)

st.markdown(
    '''
    # Serviceable Available Market
    '''
)


csv_vehicles = st.file_uploader(
    'Upload list of all vehicles in the total addressable market'
)

if csv_vehicles is not None:
    df_vehicles = pd.read_csv(csv_vehicles)
else:
    file_path_v= r"C:\Users\zwu\OneDrive - IMT Mines Albi\Documents\Data\DemandForecasting\Vehicles\Simulated_car_registration\\"+str(current_year)+"_vehicle_list_TAM.csv"
    df_vehicles = pd.read_csv(file_path_v)

df_model = df_model[['Mh', 'Cn']]

st.write(
    "Models selected:"
)

# filter geographical coverage
df_vehicles = df_vehicles.astype({'Code_commune': str})
df_vehicles = df_vehicles[df_vehicles['Code_commune'].isin(geo)]
# filter models
df_vehicles = df_vehicles.merge(df_model, how='inner', left_on = ['Mh', 'Cn'], right_on=['Mh', 'Cn'])

# group by brand and models
df_vehicles_gr_model = df_vehicles.groupby(by=['Mh', 'Cn']).count().iloc[:,:1].rename(columns={'Unnamed: 0': 'Num Vehicles'})
df_vehicles_gr_model = df_vehicles_gr_model.reset_index()


col2, col3 = st.columns([1,1])
with col2:
    st.dataframe(df_model, height=150)
    st.plotly_chart(
        px.pie(df_vehicles_gr_model, values='Num Vehicles', names='Mh', title='Distribution of Total Addressable Market among Manufacturers')
    )
with col3:
    st.metric('Number of Vehicles in the Serviceable Available Market', value=df_vehicles_gr_model['Num Vehicles'].sum())
    Mh = st.selectbox('Select a manufacturer to visualize the distribution', df_vehicles_gr_model.Mh.unique())
    df_vehicles_gr_model_filtered = df_vehicles_gr_model[df_vehicles_gr_model['Mh']==Mh]
    st.plotly_chart(
        px.pie(df_vehicles_gr_model_filtered, values='Num Vehicles', names='Cn', title='Distribution of Vehicle Models for a Selected Manufacturers')
    )

@st.cache
def convert_df(df):
    return df.to_csv().encode('utf-8')

csv = convert_df(df_vehicles)

st.download_button('Download the list of vehicles in Service Available Market here', csv, mime='text/csv', file_name=str(current_year)+'_vehicle_list_SAM.csv')




