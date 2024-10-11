import sys
sys.path.append("..")
from tools import *

'''
In this page, a standard test dataset in Occitanie Region will be detailed, which will be used in the following optimization models.

# Potential Location for facilities

'''

pop = st.slider("Select communes with population more than", min_value=10000, max_value=300000, step=1000, value=50000)
dataframe = get_communes_by_population(pop)
st.write(dataframe, len(dataframe))
fig = px.scatter_mapbox(dataframe, 
                        lat="Latitude", 
                        lon="Longitude", 
                        hover_name="Commune", 
                        hover_data=["PMUN", "COM"],
                        zoom=6,
                        height=300)

fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
st.plotly_chart(fig)

# distance matrix
dist = pd.DataFrame(index=dataframe["COM"], columns=dataframe["COM"])
for i in dist.index:
    for j in dist.columns:
        calc = haversine(dataframe.loc[dataframe["COM"]==i, "Latitude"].values[0], dataframe.loc[dataframe["COM"]==i, "Longitude"].values[0], dataframe.loc[dataframe["COM"]==j, "Latitude"].values[0], dataframe.loc[dataframe["COM"]==j, "Longitude"].values[0])
        if calc == 0:
            dist.loc[i, j] = 3
        else:
            dist.loc[i, j] = calc


st.write("Distance matrix is:", dist)

# For the moment, we are assuming that the number of vehicles to be retrofitted is the population of the commune multiplied by a coefficient.
coeff_retrofit = 0.06
parc_auto = pd.concat([dataframe["COM"], dataframe["PMUN"]*coeff_retrofit], axis=1)
parc_auto.set_index("COM", inplace=True)
parc_auto.columns = ["Demand"]

M = dataframe["COM"]

'''
# Product models to be considered
'''
# P = ["Fiat 500", "Renault Clio 3"]
P = ["Fiat 500"]
percentage_p = [0.2]
# st.write("The following vehicles will be considered: Fiat 500, Renault Clio 3.")

grid1, grid2 = np.meshgrid(M,P)
index = pd.MultiIndex.from_tuples(list(zip(grid1.flatten(), grid2.flatten())))
# We are assuming that the time period is three years, with the demand distributed by [0.2, 0.5, 0.3] for each year.
# The demand for each product in each commune is then calculated as the product of the total demand and the portion of each product.
demand = pd.DataFrame(index=index, columns=["Demand"])
for com in M:
    for p in P:
        demand.loc[(com, p), "Demand"] = parc_auto.loc[com, "Demand"]*percentage_p[P.index(p)]


T = [i for i in range(1, 21)]
p = 0.03
q = 0.38
# Initial condition: no adopters at time t=0
N0 = [0]
# Solve the differential equation
N = odeint(bass_diff, N0, [i for i in range(1, 22)], args=(p, q, 1)).flatten()


# T = [1]

dist_T = np.diff(N, prepend=0)[1:]

# dist_T = [1]

for t in T:
    demand[t] = np.floor(demand["Demand"]*dist_T[t-1])
demand.drop(columns=["Demand"], inplace=True)

'''
# Demand data
'''
def append_letter_to_first_level_index(df, letter):
    # Extract the MultiIndex
    new_index = df.index
    # Modify the first level index by appending the letter
    new_first_level = [str(item) + letter for item in new_index.get_level_values(0)]
    # Create a new MultiIndex with the modified first level
    new_multiindex = pd.MultiIndex.from_arrays([new_first_level, new_index.get_level_values(1)], names=new_index.names)
    # Set the new MultiIndex to the DataFrame
    df.index = new_multiindex
    return df

demand = append_letter_to_first_level_index(demand, "M")
st.write(demand)

x = np.linspace(1, 30, 29)
pb_EoL = 1-calculate_cum_survival_function(x)



maxcapMN = 10000
maxcapRMN = 2000
maxcapH = 15000
maxcapR = 1000
maxcapDP = 200
maxcapRF = 10000
maxcapDRU = 1000

# molMN = 100
# molRMN = 0
# molH = 150
# molR = 10
# molDP = 0
# molRF = 100
# molDRU = 0

molMN = 0
molRMN = 0
molH = 0
molR = 0
molDP = 0
molRF = 0
molDRU = 0

uofMN = 2538.2
uofRMN = 2
uofH = 1
uofR = 1.5  
uofDP = 1
uofRF = 1.5
uofDRU = 1
 
wRU = 180
wEoLPa = 70

pl_TR = 1000
tf_TR = 0.1
fr_TR = 0.8
utf_PR = 0.2
utf_RP = 0.1
lofPR = 5000
lofEoLP = 1000

af = pd.Series({'R': 100000, 'V': 500000, 'L': 200000, 'F': 700000})
of = pd.Series({'R': 5000, 'V': 25000, 'L': 10000, 'F': 35000})

Z = 1000000
D_max = 50

data = {
    "M": M,
    "P": P,
    "T": T,
    "demand": demand,
    "pb_EoL": pb_EoL,
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
    "of": of,
    "Z": Z,
    "D_max": D_max
}



with open("parameters.pkl", "wb") as f:
    pickle.dump(data, f)


