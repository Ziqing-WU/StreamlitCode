from config import *

with open("demand.pickle", "rb") as f:
    demand = pickle.load(f)

st.write("# Sets")
st.write(f"""
We consider that the potential locations for all the facilities are the same. These are selected based on the population of the communes.
""")
n = st.number_input("Enter the population threshold for the potential locations", value=10000)
df_sets = get_communes_by_population(n)
st.write(f"""
The number of potential locations for all facilities with population superior to {n} is {df_sets.shape[0]}. There distribution on the territory is shown in the following map.
""")

fig = px.scatter_mapbox(df_sets, lat="Latitude", lon="Longitude", mapbox_style="carto-positron", zoom=5.5, center = {"lat": 43.5, "lon": 2})
st.plotly_chart(fig)

st.write(f"""
$R$, $V$, $L$, $F$ are thus defined by these {df_sets.shape[0]} communes.

To simplify the model solution without loss of generality, in this case, we do not differentiate the various types of products, even if the model allows us to do so.

$P = [p]$

We suppose that the time horizon for retrofitting services to be 20 years, which is used as the planning horizon for the network design.

$T = [1, 2, 3, ..., 20]$
""")
P = ["p"]
T = [i for i in range(1, 21)]

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
fig = px.scatter_mapbox(demand_scenario, lat="Latitude", lon="Longitude", size="Number of Vehicles", mapbox_style="carto-positron", zoom=5.5, center = {"lat": 43.5, "lon": 2}, color_continuous_scale='Greys')
fig_line = px.line(x=demand_scenario.sum().index[2:-2], y=demand_scenario.sum().values[2:-2], title='Total Demand', labels={'x':'Year', 'y':'Number of Vehicles'})
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
beta = st.number_input("Enter the shape parameter of the Weibull distribution", value=6)
gamma = st.number_input("Enter the mean parameter of the Weibull distribution", value=15.2)

pb_tau = {tau:calculate_cum_distribution_function(tau, beta_shape=beta, gamma_mean = gamma) for tau in range(1, 21)}
plot_tau = px.line(x=pb_tau.keys(), y=pb_tau.values(), title='CDF of Weibull Distribution', labels={'x':'Year', 'y':'Probability'})
st.plotly_chart(plot_tau)

st.write("""
## Distance
The distance between two communes is calculated using the Haversine formula.
""")

communes = list(set(demand_scenario['code_commune_titulaire'].unique()) | set(df_sets['COM']))
demand_scenario.set_index('code_commune_titulaire', inplace=True)
dist = pd.DataFrame(index=communes, columns=communes)
for i in dist.index:
    for j in dist.columns:
        lat_i = demand_scenario.at[i, 'Latitude']
        lon_i = demand_scenario.at[i, 'Longitude']
        lat_j = demand_scenario.at[j, 'Latitude']
        lon_j = demand_scenario.at[j, 'Longitude']
        calc = haversine(lat_i, lon_i, lat_j, lon_j)
        if calc == 0:
            dist.loc[i, j] = 3
        else:
            dist.loc[i, j] = calc

st.write("Distance matrix $d_{ii'}$ is:", dist)
D_max = st.number_input("Enter the maximum allowable distance in km between a market segment and a retrofit centre", value=40)

st.write("## Weights")

wRU = st.number_input("Enter the weight of the retrofitting kit in kg", value=380)
wEoLPa = st.number_input("Enter the weight of the end-of-life parts in kg", value=150)

st.write("""## Facility Capacities
### Maximum Capacities
""")

maxcapMN = st.number_input("Enter the maximum capacity of a factory for manufacturing per planning period", value=1000)
maxcapRMN = st.number_input("Enter the maximum capacity of a factory for remanufacturing per planning period", value=500)
maxcapH = st.number_input("Enter the maximum capacity of a logistic node for handling per planning period", value=1000)
maxcapR = st.number_input("Enter the maximum capacity of a retrofit centre for retrofitting per planning period", value=40)
maxcapDP = st.number_input("Enter the maximum capacity of a retrofit centre for dismantling per planning period", value=10)
maxcapRF = st.number_input("Enter the maximum capacity of a recovery centre for refurbishing per planning period", value=100)
maxcapDRU = st.number_input("Enter the maximum capacity of a recovery centre for dismantling retrofit kits per planning period", value=50)

st.write("""### Minimum operating levels""")

molMN = st.number_input("Enter the minimum operating level of a factory for manufacturing per planning period", value=100)
molRMN = st.number_input("Enter the minimum operating level of a factory for remanufacturing per planning period", value=50)
molH = st.number_input("Enter the minimum operating level of a logistic node for handling per planning period", value=100)
molR = st.number_input("Enter the minimum operating level of a retrofit centre for retrofitting per planning period", value=4)
molDP = st.number_input("Enter the minimum operating level of a retrofit centre for dismantling per planning period", value=1)
molRF = st.number_input("Enter the minimum operating level of a recovery centre for refurbishing per planning period", value=10)
molDRU = st.number_input("Enter the minimum operating level of a recovery centre for dismantling retrofit kits per planning period", value=5)

st.write("""## Activation Footprint""")

af_F = st.number_input("Enter the amortized activation footprint of a factory per planning period", value=700000)
af_L = st.number_input("Enter the amortized activation footprint of a logistic node per planning period", value=200000)
af_R = st.number_input("Enter the amortized activation footprint of a retrofit centre per planning period", value=100000)
af_V = st.number_input("Enter the amortized activation footprint of a recovery centre per planning period", value=500000)
af = {"F": af_F, "L": af_L, "R": af_R, "V": af_V}

st.write("""## Unit Operation Footprint""")

uofMN = st.number_input("Enter the unit operation footprint of a factory for manufacturing per planning period", value=2600)
uofRMN = st.number_input("Enter the unit operation footprint of a factory for remanufacturing per planning period", value=1200)
uofH = st.number_input("Enter the unit operation footprint of a logistic node for handling per planning period", value=0.5)
uofR = st.number_input("Enter the unit operation footprint of a retrofit centre for retrofitting per planning period", value=2)
uofDP = st.number_input("Enter the unit operation footprint of a retrofit centre for dismantling per planning period", value=0.5)
uofRF = st.number_input("Enter the unit operation footprint of a recovery centre for refurbishing per planning period", value=0.4)
uofDRU = st.number_input("Enter the unit operation footprint of a recovery centre for dismantling retrofit kits per planning period", value=0.3)

st.write("""## Transportation""")
pl_TR = st.number_input("Enter the payload capacity in kg of a transport unit", value=3000)
tf_TR = st.number_input("Enter the transportation footprint in kg CO2 eq per km of a transport unit", value=0.4212) # data from ADEME for 80% payload rate
fr_TR = st.number_input("Enter the filling rate of a transport unit", value=0.8)
utf_PR = st.number_input("Enter the average transport footprint in kg CO2 eq per km for products to be retrofitted", value=0.231)
utf_RP = st.number_input("Enter the average transport footprint in kg CO2 eq per km for retrofitted products", value=0)

st.write("""## Lost Order Footprint""")
lofPR = st.number_input("Enter the lost order footprint in kg CO2 eq per product", value=27230) # we assume that the car will still be used during 10 years
lofEoLP = st.number_input("Enter the lost order footprint in kg CO2 eq per EoL product", value=5000)

Z = 1000000

data = {
    "M": demand_scenario.index,
    "R": df_sets.index,
    "V": df_sets.index,
    "L": df_sets.index,
    "F": df_sets.index,
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


if st.button("Save Parameters"):
    with open("parameters.pkl", "wb") as f:
        pickle.dump(data, f)











