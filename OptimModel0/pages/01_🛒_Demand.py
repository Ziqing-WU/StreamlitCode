import streamlit as st
import pandas as pd
import numpy.random as rd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

rd.seed(0)
st.write(
    '''
# Preamble
## Data Source
For the moment, as the data acquired from SIV is not yet available, a test dataset is generated from the public data:

[Données sur le parc de véhicules en circulation au 1er janvier 2022](https://www.statistiques.developpement-durable.gouv.fr/donnees-sur-le-parc-de-vehicules-en-circulation-au-1er-janvier-2022)

The population density data used are from INSEE: https://www.insee.fr/fr/information/2114627

## Hypotheses
The hypothesis made are stated as following:
- Only vehicles with Crit'Air $\geq$ 3 and non classé will be converted ([source](https://www.service-public.fr/particuliers/actualites/A14587))
- There is a correlation between the population density of a commune and its percentage of conversion. The initial hypothesis is proposed as below. Feel free to change if necessary.

'''
)

# Define the structure of your table
rows = ['Crit\'Air 3', 'Crit\'Air 4', 'Crit\'Air 5', 'Non classé']
columns = ['Communes denses', 'Communes de densité intermédiaire', 'Communes peu denses', 'Communes très peu denses']
initial_values = [
    [0.25, 0.20, 0.15, 0.10],
    [0.35, 0.30, 0.25, 0.20],
    [0.45, 0.40, 0.35, 0.30],
    [0.55, 0.50, 0.45, 0.40]
]

# Initialize the session state for the DataFrame if it doesn't exist
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame(initial_values, index=rows, columns=columns)

# Function to display editable table
def display_table(df, columns):
        # Create a header row for the column names
    header_cols = st.columns(len(columns) + 1)
    header_cols[0].write("")  # Empty top-left cell where row labels and column names intersect
    for i, col_name in enumerate(columns):
        header_cols[i + 1].write(f"**{col_name}**")
    for i, row in enumerate(df.index):
        cols = st.columns(len(df.columns) + 1)
        # Display the row header
        cols[0].write(f"**{row}**")
        for j, col in enumerate(df.columns):
            # Display the input fields in the rest of the columns
            with cols[j + 1]:
                st.session_state.df.at[row, col] = st.number_input("", 
                                                                   value=df.at[row, col],
                                                                   key=f"{i}-{j}",
                                                                   min_value=0.0, 
                                                                   max_value=1.0, 
                                                                   step=0.01,
                                                                   format="%.2f")

# Create and display editable table
display_table(st.session_state.df, columns)

percentage_table = st.session_state.df

percentage_table.columns = [1, 2, 3, 4]
# st.write(percentage_table)


def ratio_pop_den_critair(pop_den, critair):
    if pd.isna(pop_den) or pd.isna(critair):
        return None
    else:
        return percentage_table[pop_den][critair]

def calculate_num_retrofit(row):
    # Check for NaN in either "ratio" or '2022'
    if pd.isna(row["ratio"]) or pd.isna(row['2022']):
        return 0  # or some other appropriate default value
    else:
        return round(row["ratio"] * row['2022'])

def generate_p_q(mean, sigma):
    while True:
        p_or_q = rd.normal(mean, sigma)
        if p_or_q>0:
            return p_or_q
    
def bass(p,q, total, t):
    return total*(1 - np.exp(-(p+q)*t))/(1+q/p*np.exp(-(p+q)*t))


mean_p = 0.0019
mean_q= 0.2

st.write(r'''
- The diffusion of conversion is estimated using Bass model with parameters $p$, $q$ taken from the literature and $M$ the total market potential deduced from the exisitng ICE vehicles. The number of vehicles converted will be $N(t)=M \times \frac{1-e^{-(p+q)t}}{1+\frac{q}{p}\times e^{-(p+q)t}}$. We will use $p=0.0019$ and $q=1.2513$ ([source](https://doi.org/10.1016/j.retrec.2015.06.003)). And to add randomness in the model, we suppose that the $p$ and $q$ vary according to a normal distribution, meaning $ p \sim \mathcal{N}         '''+
         f"({mean_p}," + "\sigma_p)$ and $ q \sim \mathcal{N}"+f"({mean_q}," + " \sigma_q)$. Finally the values should be positive. If not, another sample will be generated. "
)

col1, col2 = st.columns(2)
with col1:
    sigma_p = st.number_input(r"$\sigma_p$", value=0.001, format="%.3f")
with col2:
    sigma_q = st.number_input(r"$\sigma_q$", value=0.2, format="%.3f")

st.write('''
    # Demand
         ''')

def generate_demand():
    file_path = r"C:\Users\zwu\OneDrive - IMT Mines Albi\Documents\Data\DemandForecasting\Vehicles\Parc automobile 2022\parc_vp_commune_2022_cleaned.csv"

    # population density for each commune and their mairie's location
    pop_den = pd.read_csv("pop_den.csv")
    pop_den.rename(columns={'Typo degr� de Densit�': 'Population Density'}, inplace=True)
    pop_den = pop_den[['Code Commune', 'Population Density', 'coordinates']]
    pop_den['lon'] = pop_den['coordinates'].str.strip('[]').str.split(',').str[0].astype(float)
    pop_den['lat'] = pop_den['coordinates'].str.strip('[]').str.split(',').str[1].astype(float)


    # st.dataframe(pop_den)

    parc_auto = pd.read_csv(file_path, encoding='ISO-8859-1')
    # st.dataframe(parc_auto)

    parc_auto_pop_den = pd.merge(parc_auto, pop_den, left_on='Code commune de résidence', right_on='Code Commune', how='left')[['Code commune de résidence', 'Commune de résidence', 'lon', 'lat', 'Carburant', 'Crit\'Air', '2022', 'Population Density']]

    parc_auto_pop_den["ratio"] = parc_auto_pop_den.apply(lambda row: ratio_pop_den_critair(row['Population Density'], row['Crit\'Air']), axis=1)
    parc_auto_pop_den['num_retrofit'] = parc_auto_pop_den.apply(calculate_num_retrofit, axis=1)
    parc_auto_pop_den = parc_auto_pop_den[parc_auto_pop_den['num_retrofit'] != 0]

    parc_auto_pop_den["p_q"] = [[generate_p_q(mean_p, sigma_p), generate_p_q(mean_q,sigma_q)] for i in range(parc_auto_pop_den.shape[0])]
    parc_auto_pop_den['2023'] = parc_auto_pop_den.apply(lambda row: round(bass(row['p_q'][0], row['p_q'][1], row['num_retrofit'], 1)),axis=1)
    parc_auto_pop_den['2024'] = parc_auto_pop_den.apply(lambda row: round(bass(row['p_q'][0], row['p_q'][1], row['num_retrofit'], 2)),axis=1)
    parc_auto_pop_den['2025'] = parc_auto_pop_den.apply(lambda row: round(bass(row['p_q'][0], row['p_q'][1], row['num_retrofit'], 3)),axis=1)
    parc_auto_pop_den['2026'] = parc_auto_pop_den.apply(lambda row: round(bass(row['p_q'][0], row['p_q'][1], row['num_retrofit'], 4)),axis=1)
    parc_auto_pop_den['2027'] = parc_auto_pop_den.apply(lambda row: round(bass(row['p_q'][0], row['p_q'][1], row['num_retrofit'], 5)),axis=1)
    return parc_auto_pop_den

parc_auto_pop_den = generate_demand()
# st.write(parc_auto_pop_den.head(100))

year = st.slider("Which year?", 2023, 2027, step=1)
df_year_all = parc_auto_pop_den[["Code commune de résidence", "Commune de résidence", 'lon', 'lat', "Carburant", "Crit\'Air", "num_retrofit",str(year)]]
df_year = df_year_all[df_year_all[str(year)]!=0]
# st.write(df_year)

def plot_bass_model():
    t_values = np.linspace(0,50,11)
    N_values = [bass(mean_p, mean_q, df_year_all["num_retrofit"].sum(), t) for t in t_values]
    fig = go.Figure(data=go.Scatter(x=t_values, y=N_values))
    fig.update_layout(title='Bass Model', xaxis_title='Time (year)', yaxis_title='Number of Vehicles Converted')
    st.plotly_chart(fig)
                    
plot_bass_model()

fig = px.scatter_mapbox(
    df_year, 
    lat='lat',  # Replace with your actual latitude column name
    lon='lon',  # Replace with your actual longitude column name
    color='Crit\'Air',  # Column to be used for color coding
    size=str(year),  # Column to be used for sizing points
    hover_name='Commune de résidence',  # Column to show on hover
    hover_data=['Carburant'],
    zoom=4,  # Initial zoom level
    center={"lat": 46.2276, "lon": 2.2137},  # Center of the map (France coordinates)
    title=f"Scatter Map for Year {year}"
)

df_year_all = df_year_all.to_csv()
st.plotly_chart(fig)
<<<<<<< HEAD:OptimModel0/pages/01_🛒_Demand.py
st.write(f"Number of demand points is {len(df_year['Code commune de résidence'].unique())} in {year}")
st.write(f"Total number of vehicles converted is {df_year[str(year)].sum()} in {year}")
=======
st.write(f"Number of demand points is {len(df_year['Code commune de résidence'].unique())}")
>>>>>>> parent of ff9b015 (uncovered demand shown):OptimModel/pages/01_🛒_Demand.py
st.download_button(
    label="Download demand data as CSV",
    data=df_year_all,
    file_name='df_year_all.csv',
    mime='text/csv',
)