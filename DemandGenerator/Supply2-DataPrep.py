from config import *

st.title('Data Preparation')

with st.sidebar:
    """
    Navigation

    [Introduction](#introduction)

    [Sets](#sets)

    [Parameters](#parameters)
    - [Demand](#demand)
    - [Distance](#distance)
    - [Weights](#weights)
    - [Facility Capacities](#facility-capacities)
    - [Activation Footprint](#activation-footprint)
    - [Unit Operation Footprint](#unit-operation-footprint)
    - [Transportation](#transportation)
    - [Lost Order Footprint](#lost-order-footprint)
    """

with open("demand.pickle", "rb") as f:
    demand = pickle.load(f)

collab = st.radio("Select the collaborative strategy", ["Integrated", "Together", "Hyperconnected"], index=0)
st.write("# Sets")
st.write(f"""
We consider that the potential locations for all the facilities are the same. These are selected based on the population of the communes.
""")
n = st.number_input("Enter the population threshold for the potential locations", value=20000, step=1000)
df_sets = get_communes_by_population(n)
st.write(f"""
The number of potential locations for all facilities with population superior to {n} is {df_sets.shape[0]}. There distribution on the territory is shown in the following map.
""")

fig = px.scatter_mapbox(df_sets, lat="Latitude", lon="Longitude", hover_data = 'COM', mapbox_style="carto-positron", zoom=5.5, center = {"lat": 43.5, "lon": 2})
st.plotly_chart(fig)

R = get_communes_by_population(n)["COM"]
R = st.multiselect("Select the potential locations for the retrofit centers", R, default = ["11069","12145","12202","30189","31557","32013","34301","65440","66136","81004","82121"])
if st.button("Show selected locations", key="R"):
    fig = px.scatter_mapbox(df_sets[df_sets['COM'].isin(R)], lat="Latitude", lon="Longitude", hover_data = 'COM', mapbox_style="carto-positron", zoom=5.5, center = {"lat": 43.5, "lon": 2})
    st.plotly_chart(fig)

F = st.multiselect("Select the potential locations for the factories, logistics nodes, and recovery centers", df_sets["COM"].values, default = ["30189","34032", "31555", "66136", "81065"])
if st.button("Show selected locations", key="F"):
    fig = px.scatter_mapbox(df_sets[df_sets['COM'].isin(F)], lat="Latitude", lon="Longitude", hover_data = 'COM', mapbox_style="carto-positron", zoom=5.5, center = {"lat": 43.5, "lon": 2})
    st.plotly_chart(fig)

L = F 
V = F


st.write(f"""
To simplify the model solution without loss of generality, in this case, we do not differentiate the various types of products, even if the model allows us to do so.

$P = [p]$

We suppose that the time horizon for retrofitting services to be 20 years, which is used as the planning horizon for the network design.

$T = [1, 2, 3, ..., 20]$ for integrated and together scenario

$T = [1, 2, 3, ..., 240]$ for hyperconnected scenario
""")
P = ["p"]
T = [i for i in range(1, 21)]
if collab == "Hyperconnected":
    T = [i for i in range(1, 241)]

map_df = pd.read_csv(rf'{executive_factor_folder}\Ref-1-CodeGeo\GeoPosition.csv',
                     dtype={
                         'Code Commune': 'object',
                         'Longitude': 'float64',
                         'Latitude': 'float64',
                         # Add other columns and their data types here
                     }
                    ).set_index("Code Commune")

for key in demand.keys():
    demand[key] = demand[key].merge(map_df, left_on='code_commune_titulaire', right_index=True)
    demand[key] = demand[key].rename(columns = {"retrofit_service": "Number of Vehicles"})

st.write("""
# Parameters
## Demand
""")

scenario = st.selectbox("Select the demand scenario to visualize", list(demand.keys()), index=0)
demand_scenario = demand[scenario]
st.write(f"The number of demand points is {len(demand_scenario)}")

if collab == "Hyperconnected":
    # monthly_weights = np.array([7.0, 7.0, 7.5, 8.0, 8.0, 8.0, 
    #                                7.5, 7.5, 8.0, 8.0, 8.0, 7.0])
    monthly_weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
                                   1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    monthly_weights = monthly_weights / monthly_weights.sum()
    annual_columns = [i for i in range(1, 21)]
    # Melt the dataframe to long format for annual demands
    demand_scenario = demand_scenario.melt(id_vars=['code_commune_titulaire', 'Number of Vehicles', 
                            'Longitude', 'Latitude'], 
                    value_vars=annual_columns,
                    var_name='Year', 
                    value_name='Annual_Demand')

    # Convert 'Year' to integer
    demand_scenario['Year'] = demand_scenario['Year'].astype(int)
    # Repeat each row 12 times for 12 months
    demand_scenario_expanded = demand_scenario.loc[demand_scenario.index.repeat(12)].reset_index(drop=True)
    
    # Assign month numbers (1 to 12)
    demand_scenario_expanded['Month'] = np.tile(np.arange(1, 13), len(demand_scenario))

    # Assign corresponding monthly weights
    demand_scenario_expanded['Monthly_Weight'] = np.tile(monthly_weights, len(demand_scenario))

    # Calculate monthly demand
    demand_scenario_expanded['Monthly_Demand'] = demand_scenario_expanded['Annual_Demand'] * demand_scenario_expanded['Monthly_Weight']

    # Floor the monthly demands
    demand_scenario_expanded['Monthly_Demand_Floor'] = np.floor(demand_scenario_expanded['Monthly_Demand']).astype(int)


    # Calculate the remaining demand to distribute
    demand_scenario_expanded['Remaining'] = demand_scenario_expanded['Annual_Demand'] - demand_scenario_expanded.groupby(
        ['code_commune_titulaire', 'Year'])['Monthly_Demand_Floor'].transform('sum')

    # Calculate the fractional parts
    demand_scenario_expanded['Fraction'] = demand_scenario_expanded['Monthly_Demand'] - np.floor(demand_scenario_expanded['Monthly_Demand'])

    # Step 4: Sort the dataframe to prioritize months with higher fractional parts
    np.random.seed(42)  # For reproducibility; remove or change the seed as needed
    demand_scenario_expanded['RandomOrder'] = np.random.rand(len(demand_scenario_expanded))
    demand_scenario_expanded = demand_scenario_expanded.sort_values(
    by=['code_commune_titulaire', 'Year', 'Fraction', 'RandomOrder'],
    ascending=[True, True, False, True]  # 'Fraction' descending, 'RandomOrder' ascending
    ).reset_index(drop=True)
    demand_scenario_expanded = demand_scenario_expanded.drop(columns=['RandomOrder'])
    # Step 5: Assign the remaining demand
    def distribute_remaining(group):
        remaining = group['Remaining'].iloc[0]
        remaining = int(min(remaining, len(group)))
        if remaining > 0:
            group.loc[group.index[:remaining], 'Monthly_Demand_Floor'] += 1
        return group
    demand_scenario_expanded = demand_scenario_expanded.groupby(['code_commune_titulaire', 'Year']).apply(distribute_remaining).reset_index(drop=True)
    # Step 6: Rename the final monthly demand column
    demand_scenario_expanded['Monthly_Demand_Final'] = demand_scenario_expanded['Monthly_Demand_Floor']

    # Drop intermediate columns
    demand_scenario = demand_scenario_expanded.drop(['Monthly_Weight', 'Monthly_Demand', 'Monthly_Demand_Floor', 'Remaining', 'Fraction'], axis=1)
    demand_scenario["Month_Count"] = (demand_scenario["Year"]-1) * 12 + demand_scenario["Month"]

    demand_scenario = demand_scenario.pivot_table(
    index=['code_commune_titulaire', 'Number of Vehicles', 'Longitude', 'Latitude'],
    columns='Month_Count',
    values='Monthly_Demand_Final',
    aggfunc='sum'  # or 'mean', 'max', etc., depending on your needs
    ).reset_index()

st.write(demand_scenario)
fig = px.scatter_mapbox(demand_scenario, lat="Latitude", lon="Longitude", size="Number of Vehicles", mapbox_style="carto-positron", zoom=5.5, center = {"lat": 43.5, "lon": 2}, color_continuous_scale='Greys')

map_R = df_sets[df_sets['COM'].isin(R)]
map_F = df_sets[df_sets['COM'].isin(F)]
st.write(map_R)
fig.add_trace(go.Scattermapbox(
    lat=map_F["Latitude"],
    lon=map_F["Longitude"],
    mode='markers',
    marker=go.scattermapbox.Marker(size=12, color='yellow'),
    name='Potential Locations <br> for Factories, <br>Logistics Nodes, <br>and Recovery Centers',
    showlegend=True
))

fig.add_trace(go.Scattermapbox(
    lat=map_R["Latitude"],
    lon=map_R["Longitude"],
    mode='markers',
    marker=go.scattermapbox.Marker(size=8, color='red'),
    name='Potential Locations <br> for Retrofit Centers',
    showlegend=True
))




if collab == "Integrated" or collab == "Together":
    x_title = "Year"
else:
    x_title = "Month"
fig_line = px.line(x=demand_scenario.sum().loc[T].index, y=demand_scenario.sum().loc[T].values, title='Total Demand', labels={'x':x_title, 'y':'Number of Vehicles'})

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig_line)
with col2:
    st.plotly_chart(fig)

st.write("""
The probability that a retrofitted product reaches its end of life after year $\\tau$ is given by the cumulative distribution function of the Weibull distribution with parameters proposed by Held et al. 2021.
We assume that the lifecycle of the retrofitted vehicles follows the same pattern as the new average vehicle in the French market.
However the parameters can be adjusted here as well.
""")

if collab == "Integrated" or collab == "Together":
    beta = st.number_input("Enter the shape parameter of the Weibull distribution", value=6)
    gamma = st.number_input("Enter the mean parameter (in years) of the Weibull distribution", value=15.2)
else:
    beta = st.number_input("Enter the shape parameter of the Weibull distribution", value=6)
    gamma = st.number_input("Enter the mean parameter (in months) of the Weibull distribution", value=15.2*12)

pb_tau = {tau:calculate_cum_distribution_function(tau, beta_shape=beta, gamma_mean = gamma) for tau in T}
plot_tau = px.line(x=pb_tau.keys(), y=pb_tau.values(), title='CDF of Weibull Distribution', labels={'x':x_title, 'y':'Probability'})
st.plotly_chart(plot_tau)


st.write("""
## Distance
The distance between two communes is calculated using the Haversine formula.
""")

@st.cache_data
def compute_distance_matrix(demand_scenario: pd.DataFrame, df_sets: pd.DataFrame) -> pd.DataFrame:
    # Combine unique commune codes from both DataFrames
    communes = list(
        set(demand_scenario['code_commune_titulaire'].unique()) | set(df_sets['COM'])
    )
    
    # Set 'code_commune_titulaire' as the index for demand_scenario
    demand_scenario = demand_scenario.set_index('code_commune_titulaire')
    
    # Initialize the distance DataFrame with communes as both index and columns
    dist = pd.DataFrame(index=communes, columns=communes, dtype=float)
    
    # Iterate over each pair of communes to calculate distances
    for i in dist.index:
        for j in dist.columns:
            lat_i = demand_scenario.at[i, 'Latitude']
            lon_i = demand_scenario.at[i, 'Longitude']
            lat_j = demand_scenario.at[j, 'Latitude']
            lon_j = demand_scenario.at[j, 'Longitude']
            
            # Calculate the haversine distance
            distance = haversine(lat_i, lon_i, lat_j, lon_j)
            
            # Assign distance or 3 if the distance is zero
            dist.at[i, j] = 3 if distance == 0 else distance
    
    return dist

dist = compute_distance_matrix(demand_scenario, df_sets)

st.write("Distance matrix $d_{ii'}$ is:", dist)
D_max = st.number_input("Enter the maximum allowable distance in km between a market segment and a retrofit centre", value=60)

st.write("## Weights")

wRU = st.number_input("Enter the weight of the retrofitting kit in kg", value=380)
wEoLPa = st.number_input("Enter the weight of the end-of-life parts in kg", value=150)

st.write("""## Facility Capacities
### Maximum Capacities
""")

if collab == "Hyperconnected":
    maxcapMN = st.number_input("Enter the maximum capacity of a factory for manufacturing per planning period", value=11000/12)
    maxcapRMN = st.number_input("Enter the maximum capacity of a factory for remanufacturing per planning period", value=2200/12)
    maxcapH = st.number_input("Enter the maximum capacity of a logistic node for handling per planning period", value=11000/12)
    maxcapR = st.number_input("Enter the maximum capacity of a retrofit centre for retrofitting per planning period", value=5000/12)
    maxcapDP = st.number_input("Enter the maximum capacity of a retrofit centre for dismantling per planning period", value=1000/12)
    maxcapRF = st.number_input("Enter the maximum capacity of a recovery centre for refurbishing per planning period", value=3000/12)
    maxcapDRU = st.number_input("Enter the maximum capacity of a recovery centre for dismantling retrofit kits per planning period", value=1000/12)
else:
    maxcapMN = st.number_input("Enter the maximum capacity of a factory for manufacturing per planning period", value=11000)
    maxcapRMN = st.number_input("Enter the maximum capacity of a factory for remanufacturing per planning period", value=2200)
    maxcapH = st.number_input("Enter the maximum capacity of a logistic node for handling per planning period", value=11000)
    maxcapR = st.number_input("Enter the maximum capacity of a retrofit centre for retrofitting per planning period", value=5000)
    maxcapDP = st.number_input("Enter the maximum capacity of a retrofit centre for dismantling per planning period", value=1000)
    maxcapRF = st.number_input("Enter the maximum capacity of a recovery centre for refurbishing per planning period", value=3000)
    maxcapDRU = st.number_input("Enter the maximum capacity of a recovery centre for dismantling retrofit kits per planning period", value=1000)

st.write("""### Minimum operating levels""")

molMN = st.number_input("Enter the minimum operating level of a factory for manufacturing per planning period", value=0)
molRMN = st.number_input("Enter the minimum operating level of a factory for remanufacturing per planning period", value=0)
molH = st.number_input("Enter the minimum operating level of a logistic node for handling per planning period", value=0)
molR = st.number_input("Enter the minimum operating level of a retrofit centre for retrofitting per planning period", value=0)
molDP = st.number_input("Enter the minimum operating level of a retrofit centre for dismantling per planning period", value=0)
molRF = st.number_input("Enter the minimum operating level of a recovery centre for refurbishing per planning period", value=0)
molDRU = st.number_input("Enter the minimum operating level of a recovery centre for dismantling retrofit kits per planning period", value=0)

st.write("""## Activation Footprint""")
if collab == "Hyperconnected":
    af_F = st.number_input("Enter the amortized activation footprint of a factory per planning period", value=160000/12)
    af_L = st.number_input("Enter the amortized activation footprint of a logistic node per planning period", value=50000/12)
    af_R = st.number_input("Enter the amortized activation footprint of a retrofit centre per planning period", value=20000/12)
    af_V = st.number_input("Enter the amortized activation footprint of a recovery centre per planning period", value=100000/12)
else:
    af_F = st.number_input("Enter the amortized activation footprint of a factory per planning period", value=160000)
    af_L = st.number_input("Enter the amortized activation footprint of a logistic node per planning period", value=50000)
    af_R = st.number_input("Enter the amortized activation footprint of a retrofit centre per planning period", value=20000)
    af_V = st.number_input("Enter the amortized activation footprint of a recovery centre per planning period", value=100000)




af = {"F": af_F, "L": af_L, "R": af_R, "V": af_V}

st.write("""## Unit Operation Footprint""")

uofMN = st.number_input("Enter the unit operation footprint of a factory for manufacturing", value=600)
uofRMN = st.number_input("Enter the unit operation footprint of a factory for remanufacturing", value=200)
uofH = st.number_input("Enter the unit operation footprint of a logistic node for handling", value=0.5)
uofR = st.number_input("Enter the unit operation footprint of a retrofit centre for retrofitting", value=2)
uofDP = st.number_input("Enter the unit operation footprint of a retrofit centre for dismantling", value=0.5)
uofRF = st.number_input("Enter the unit operation footprint of a recovery centre for refurbishing", value=0.4)
uofDRU = st.number_input("Enter the unit operation footprint of a recovery centre for dismantling retrofit kits", value=0.3)

st.write("""## Transportation""")
pl_TR = st.number_input("Enter the payload capacity in kg of a transport unit", value=3000)
tf_TR = st.number_input("Enter the transportation footprint in kg CO2 eq per km of a transport unit", value=0.6) # 0.4242*1.4  data from ADEME for 80% payload rate
fr_TR = st.number_input("Enter the filling rate of a transport unit", value=0.8)
utf_PR = st.number_input("Enter the average transport footprint in kg CO2 eq per km for products to be retrofitted", value=0.231)
utf_RP = st.number_input("Enter the average transport footprint in kg CO2 eq per km for retrofitted products", value=0)

st.write("""## Lost Order Footprint""")
lofPR = st.number_input("Enter the lost order footprint in kg CO2 eq per product", value=11000) # we assume that the car will still be used during 5 years
lofEoLP = st.number_input("Enter the lost order footprint in kg CO2 eq per EoL product", value=500)

Z = 5000
data = {
    "collab": collab,
    "M": demand_scenario["code_commune_titulaire"].unique(),
    "R": R,
    "V": V,
    "L": L,
    "F": F,
    "P": P,
    "T": T,
    "demand": demand_scenario,
    "pb_EoL": pb_tau,
    "dist": dist,
    "maxcapMN": maxcapMN,
    "maxcapRMN": maxcapRMN,
    "maxcapH": maxcapH,
    "maxcapR": maxcapR,
    "maxcapDP": maxcapDP,
    "maxcapRF": maxcapRF,
    "maxcapDRU": maxcapDRU,
    "molMN": molMN,
    "molRMN": molRMN,
    "molH": molH,
    "molR": molR,
    "molDP": molDP,
    "molRF": molRF,
    "molDRU": molDRU,
    "uofMN": uofMN,
    "uofRMN": uofRMN,
    "uofH": uofH,
    "uofR": uofR,
    "uofDP": uofDP,
    "uofRF": uofRF,
    "uofDRU": uofDRU,
    "wRU": wRU,
    "wEoLPa": wEoLPa,
    "pl_TR": pl_TR,
    "tf_TR": tf_TR,
    "fr_TR": fr_TR,
    "utf_PR": utf_PR,
    "utf_RP": utf_RP,
    "lofPR": lofPR,
    "lofEoLP": lofEoLP,
    "af": af,
    "Z": Z,
    "D_max": D_max
}

# if st.button("Save Parameters"):
#     with open("parameters.pkl", "wb") as f:
#         pickle.dump(data, f)
#     st.success("Parameters saved successfully.")
#     st.write("File saved at:", os.path.abspath("parameters.pkl"))

if st.button("Save Parameters"):
    with open(f"parameters_{collab}_{scenario}_r{len(R)}_f{len(F)}.pkl", "wb") as f:
        pickle.dump(data, f)
    st.success("Parameters saved successfully.")
    st.write("File saved at:", os.path.abspath(f"parameters_{collab}_{scenario}_r{len(R)}_f{len(F)}.pkl"))









