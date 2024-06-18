import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import math
from gurobipy import Model, GRB, quicksum

st.markdown('<a name="illustration"></a>', unsafe_allow_html=True)
st.header("Illustration")
video_file = open(r"C:\Users\zwu\OneDrive - IMT Mines Albi\Documents\Data\ModelOptim\Video\atomistic.mp4", 'rb')
video_bytes = video_file.read()
st.video(video_bytes)

pop = pd.read_csv(r"C:\Users\zwu\OneDrive - IMT Mines Albi\Documents\Data\ModelOptim\General\ensemble_population_2021\donnees_communes.csv", sep=";")[["COM", "Commune", "PMUN"]].sort_values(by="PMUN", ascending=False)
mappings = {
    ('69381', '69389'): ['69123', "Lyon"],
    ('75101', '75120'): ['75056', "Paris"], 
    ('13201', '13216'): ['13055', "Marseille"]
    }

def map_code_arrondissement(row):
    for code_range, mapped_value in mappings.items():
        if code_range[0] <= row["COM"] <= code_range[1]:
            return pd.Series([mapped_value[0], mapped_value[1]])
    return pd.Series([row['COM'], row['Commune']])
pop = pop[pop["PMUN"] > 10000]
pop[['COM', 'Commune']] = pop.apply(map_code_arrondissement, axis=1)

pop = pop.groupby(["COM", "Commune"]).sum().sort_values(by="PMUN", ascending=False).reset_index()


coordinates = pd.read_csv(r"C:\Users\zwu\OneDrive - IMT Mines Albi\Documents\Data\ModelOptim\General\pop_den.csv")[["Code Commune", "coordinates"]]

commune_10k = pd.merge(pop, coordinates, left_on="COM", right_on="Code Commune", how="left")[["COM", "Commune", "PMUN", "coordinates"]]
commune_10k['lon'] = commune_10k['coordinates'].str.strip('[]').str.split(',').str[0].astype(float)
commune_10k['lat'] = commune_10k['coordinates'].str.strip('[]').str.split(',').str[1].astype(float)   
commune_10k = commune_10k.set_index("COM")
commune_10k = commune_10k[~commune_10k.index.str.startswith("97")]

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = R * c
    return d

# distance_matrix = pd.DataFrame(index=commune_10k["COM"], columns=commune_10k["COM"])
# t=0
# total = len(commune_10k["COM"])**2
# for i in commune_10k["COM"]:
#     for j in commune_10k["COM"]:
#         distance_matrix.loc[i, j] = haversine(commune_10k[commune_10k["COM"] == i]["lat"].values[0], commune_10k[commune_10k["COM"] == i]["lon"].values[0], commune_10k[commune_10k["COM"] == j]["lat"].values[0], commune_10k[commune_10k["COM"] == j]["lon"].values[0])
#         t+=1
#         if t%10 == 0:
#             st.write(f"Progress: {t/total*100}%")

# st.dataframe(distance_matrix)
# distance_matrix.to_csv(r"C:\Users\zwu\OneDrive - IMT Mines Albi\Documents\Data\ModelOptim\General\distance_matrix.csv")

distance_matrix = pd.read_csv(r"C:\Users\zwu\OneDrive - IMT Mines Albi\Documents\Data\ModelOptim\General\distance_matrix.csv", index_col=0)

st.markdown('<a name="model"></a>', unsafe_allow_html=True)
st.header("Model")
st.write(
    r'''
### Hypotheses
- All types of products have the same weight
### Indices and sets
- $m$ market segment
- $M$ set of all market segments
- $r$ retrofit center
- $R$ set of all potential retrofit centers
- $v$ recovery center
- $V$ set of recovery centers
- $l$ logistics node
- $L$ set of all potential logistics nodes
- $f$ factory
- $F$ set of all potential factories
- $p$ product model
- $P$ set of all product models

### Decision variables
- $x_{r}$ binary variable equals to 1 if retrofit center $r$ is open
- $x_{v}$ binary variable equals to 1 if recovery center $v$ is open
- $x_{l}$ binary variable equals to 1 if logistics node is $l$ is open
- $x_{f}$ binary variable equals to 1 if factory $f$ is open
- $q_{flp}$ volume of retrofit units of product model $p$ shipped from factory $f$ to logistics node $l$
- $q_{lrp}$ volume of retrofit units of product model $p$ shipped from logistics node $l$ to retrofit center $r$
- $q_{rmp}$ volume of retrofitted products with original model $p$ shipped from retrofit center $r$ to market segment $m$ 
- $q_{rvp}$ volume of extracted parts after retrofit operation with model $p$ shipped from retrofit center $r$ to recovery center $v$ 
- $q_{vlp}$ volume of refurbished retrofit units with original model $p$ from recovery center $v$ to logistics node $l$ 
- $q_{vfp}$ volume of refurbished products with original model $p$ from recovery center $v$ to factory $f$ 
- $q_{mvp}$ volume of end-of-life products with original model $p$ from market segment $m$ to recovery center $v$

### Parameters
- $w{ru}$ retrofit unit weight in kg
- $w{rpd}$ retrofitted product weight in kg
- $w{eolpd}$ End-of-life product weight in kg
- $w{rtp}$ retrieved part weight in kg
- $w{rp}$ refurbished part weight in kg
- $tc$ unit cost of shipping 1 kg product per km
- $d_{fl}, d_{lr}, d_{rm}, d_{mv}, d_{rv}, d_{vl}, d_{vf}$ distance in km from factory $f$ to logistics node $l$, from logistics node $l$ to retrofit center $r$, from retrofit center $r$ to market segment $m$, from market segment $m$ to recovery center $v$, from retrofit center $r$ to recovery center $v$, from recovery center $v$ to logistics node $l$, from recovery center $v$ to factory $f$
- $ic_{f}, ic_{l}, ic_{r}, ic_{v}$ installation cost for factory $f$, logistics node $l$, retrofit center $r$, and recovery center $v$
- $dm_{mp}$ retrofit demand of vehicle model $p$ at market segment $m$
- $dmeol_{mp}$ EoL treatement demand of vehicle model $p$ at market segment $m$

### Objective functions
MIN Total Cost = Transportation Cost (TC) + Installation Cost (IC) 


''')
st.latex(r'''         
\begin{aligned}
\text{Transportation Cost} = tc \times  
&[\sum_{f \in F} \sum_{l \in L} \sum_{p \in P} q_{flp} \cdot wru \cdot d_{fl} + \\
&\sum_{l \in L} \sum_{r \in R} \sum_{p \in P} q_{lrp} \cdot wru \cdot d_{lr} + \\
&\sum_{r \in R} \sum_{m \in M} \sum_{p \in P} q_{rmp} \cdot wrpd \cdot d_{rm} + \\
&\sum_{r \in R} \sum_{v \in V} \sum_{p \in P} q_{rvp} \cdot wrtp \cdot d_{rv} + \\
&\sum_{v \in V} \sum_{l \in L} \sum_{p \in P} q_{vlp} \cdot wru \cdot d_{vl} + \\
&\sum_{v \in V} \sum_{f \in F} \sum_{p \in P} q_{vfp} \cdot wrp \cdot d_{vf} + \\
&\sum_{m \in M} \sum_{v \in V} \sum_{p \in P} q_{mvp} \cdot weolpd \cdot d_{mv}] \\
\end{aligned}
''')

st.write(r'''
$$\text{Installation Cost} = \sum_{f \in F} ic_{f} \cdot x_{f} + \sum_{l \in L} ic_{l} \cdot x_{l} + \sum_{r \in R} ic_{r} \cdot x_{r} + \sum_{v \in V} ic_{v} \cdot x_{v}$$

### Constraints
- Demand fulfillment for retrofitted and EoL products:
    - $\sum_{r \in R} q_{rmp} = dm_{mp} \quad \forall m \in M, \forall p \in P$
    - $\sum_{v \in V} q_{mvp} = dmeol_{mp} \quad \forall m \in M, \forall p \in P$

- Facility operation constraints: (with $M$ a large enough constant)
    - Transportation can only occur from an open factory to an open logistics node
        - $q_{flp} \leq M \cdot x_{f} \quad \forall f \in F, \forall l \in L, \forall p \in P$
        - $q_{flp} \leq M \cdot x_{l} \quad \forall f \in F, \forall l \in L, \forall p \in P$
    - Transportation can only occur from an open logistics node to an open retrofit center
        - $q_{lrp} \leq M \cdot x_{l} \quad \forall l \in L, \forall r \in R, \forall p \in P$
        - $q_{lrp} \leq M \cdot x_{r} \quad \forall l \in L, \forall r \in R, \forall p \in P$
    - Transportation can only occur from an open retrofit center to the market
        - $q_{rmp} \leq M \cdot x_{r} \quad \forall r \in R, \forall m \in M, \forall p \in P$
    - Transportation can only occur from an open retrofit center to an open recovery center
        - $q_{rvp} \leq M \cdot x_{r} \quad \forall r \in R, \forall v \in V, \forall p \in P$
        - $q_{rvp} \leq M \cdot x_{v} \quad \forall r \in R, \forall v \in V, \forall p \in P$
    - Transportation can only occur from an open recovery center to an open logistics node
        - $q_{vlp} \leq M \cdot x_{v} \quad \forall v \in V, \forall l \in L, \forall p \in P$
        - $q_{vlp} \leq M \cdot x_{l} \quad \forall v \in V, \forall l \in L, \forall p \in P$
    - Transportation can only occur from an open recovery center back to an open factory
        - $q_{vfp} \leq M \cdot x_{v} \quad \forall v \in V, \forall f \in F, \forall p \in P$
        - $q_{vfp} \leq M \cdot x_{f} \quad \forall v \in V, \forall f \in F, \forall p \in P$
    - Transportation can only occur from the market to an open recovery center
        - $q_{mvp} \leq M \cdot x_{v} \quad \forall m \in M, \forall v \in V, \forall p \in P$
- Flow conservation constraints:
    - All refurbished parts are used in the manufacturing of new retrofit kits
        - $\sum_{l\in L} q_{flp} \geq \sum_{v\in V} q_{vfp} \quad \forall f \in F, \forall p \in P$
    - For a logistics node, parts that are sending out to retrofit centers are from factory or recovery center
        - $\sum_{f \in F} q_{flp} + \sum_{v \in V} q_{vlp} = \sum_{r \in R} q_{lrp} \quad \forall l \in L, \forall p \in P$
    - For a retrofit center, the products can only be retrofitted if the retrofit units are acquired 
        - $\sum_{l \in L} q_{lrp} = \sum_{m \in M} q_{rmp} \quad \forall r \in R, \forall p \in P$
    - For a retrofit center, the volume of retrieved parts equals to the volume of retrofitted product
        - $\sum_{v \in V} q_{rvp} = \sum_{m \in M} q_{rmp} \quad \forall r \in R, \forall p \in P$
    - For a recovery center, the volume of refurbished parts and refurbished retrofit units should not exceed the volume of End-of-life products that are taken back
        - $\sum_{l \in L}q_{vlp} + \sum_{f\in F}q_{vfp}\leq \sum_{m\in M}q_{mvp} \quad \forall v \in V, \forall p \in P$
- Non-negativity and binary constraints
    - $x_{f}, x_{l}, x_{r}, x_{v} \in \{0, 1\}$
    - $q_{flp}, q_{lrp}, q_{rmp}, q_{rvp}, q_{vlp}, q_{vfp}, q_{mvp} \geq 0$
'''
)

st.write('''
         # Instances
         ## Parameters
         ''')
wru = st.number_input('Retrofit unit weight in kg (wru)', value=300)
wrpd = st.number_input('Retrofitted product weight in kg (wrpd)', value=1300)
weolpd = st.number_input('End-of-life product weight in kg (weolpd)', value=1300)
wrtp = st.number_input('Retrieved part weight in kg (wrtp)', value=220)
wrp = st.number_input('Refurbished part weight in kg (wrp)', value=180)
tc = st.number_input('Unit cost of shipping per kg per km (tc)', min_value=0.0, max_value=1.0, value=0.05, format='%f')
ic_f_min, ic_f_max = st.slider('Factory installation cost range', 50000, 1000000, (190000, 220000))
ic_l_min, ic_l_max = st.slider('Logistics node installation cost range', 50000, 1000000, (100000, 150000))
ic_r_min, ic_r_max = st.slider('Retrofit center installation cost range', 50000, 1000000, (50000, 100000))
ic_v_min, ic_v_max = st.slider('Recovery center installation cost range', 50000, 1000000, (200000, 250000))


st.write('''
         ## Sets
            ''')

nb_p = st.number_input('Number of product models (P)', value=2)
nb_f = st.number_input('Number of potential factories (F)', value=10)
nb_l = st.number_input('Number of potential logistics nodes (L)', value=20)
nb_r = st.number_input('Number of potential retrofit centers (R)', value=50)
nb_v = st.number_input('Number of potential recovery centers (V)', value=10)
nb_m = st.number_input('Number of market segments (M)', value=100)


M = commune_10k.head(nb_m)
P=["p"+str(i+1) for i in range(nb_p)]
R = commune_10k.head(nb_r*10).sample(nb_r, random_state=0)
L = commune_10k.head(nb_l*10).sample(nb_l, random_state=1)
F = commune_10k.head(nb_f*10).sample(nb_f, random_state=2)
V = commune_10k.head(nb_v*10).sample(nb_v, random_state=3)

df = pd.concat([M.assign(Type='Market'), R.assign(Type='Retrofit Center'), L.assign(Type='Logistics Node'), F.assign(Type='Factory'), V.assign(Type='Recovery Center')])
df.reset_index(inplace=True)

installation_cost = pd.DataFrame(columns=["Type", "Location", "Installation Cost"])
installation_cost["Type"] = ["Factory" for f in F.index] + ["Logistics Node" for l in L.index] + ["Retrofit Center" for r in R.index] + ["Recovery Center" for v in V.index]
installation_cost["Location"] = F.index.to_list() + L.index.to_list() + R.index.to_list() + V.index.to_list()
np.random.seed(0)
installation_cost["Installation Cost"] = np.concatenate([np.random.randint(ic_f_min, high=ic_f_max, size=len(F)), np.random.randint(ic_l_min, high=ic_l_max, size=len(L)), np.random.randint(ic_r_min, high=ic_r_max, size=len(R)), np.random.randint(ic_v_min, high=ic_v_max, size=len(V))])
dm_mp = pd.DataFrame(columns= P, index = M.index)


pop = pop.set_index("COM")
def retrofit_demand(row, range = 0.04, base = 0.01):
    pmun = pop.loc[row.name, "PMUN"]
    rand = np.random.rand(len(P))*range+base
    return np.round(pmun*rand)
dm_mp[P] = dm_mp.apply(lambda x:pd.Series(retrofit_demand(x)), axis=1)

dmeol_mp = pd.DataFrame(columns= P, index = M.index)
dmeol_mp[P] = dm_mp.apply(lambda x:pd.Series(retrofit_demand(x, range=0.02, base=0.001)), axis=1)

big_M = 100000000

# st.write(df)

# type_to_icon = {
#     'Factory': 'industrial',
#     'Logistics Node': 'warehouse',
#     'Retrofit Center': 'car-repair',
#     'Recovery Center': 'recycling',
#     'Market': 'shop'
# }

type_to_icon = {
    'Factory': 'circle',
    'Logistics Node': 'circle',
    'Retrofit Center': 'circle',
    'Recovery Center': 'circle',
    'Market': 'circle'
}
type_to_color = {
    'Factory': '#ED1C24',
    'Logistics Node': '#488D25',
    'Retrofit Center': '#020100',
    'Recovery Center': '#F1D302',
    'Market': '#235789'
}

fig = go.Figure()

# Loop through each facility type and add a scattermapbox trace for each
for facility_type, group_df in df.groupby('Type'):
    fig.add_trace(go.Scattermapbox(
        lat=group_df['lat'],
        lon=group_df['lon'],
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=10,
            opacity=0.7,
            symbol=type_to_icon[facility_type],
            color=type_to_color[facility_type]
        ),
        name=facility_type
    ))

# Update layout with Mapbox access token and style
fig.update_layout(
    mapbox_style="open-street-map",  # or use "open-street-map" if you don't have a Mapbox token
    mapbox_zoom=3.5,
    mapbox_center={"lat": df['lat'].median(), "lon": df['lon'].median()}
)

st.markdown('<a name="potential-locations"></a>', unsafe_allow_html=True)
st.header("Potential Facility Locations")
# Show the figure
st.plotly_chart(fig)

st.markdown('<a name="results"></a>', unsafe_allow_html=True)
st.header("Results")


model = Model("SupplyChainDesign1")
F = F.index.to_list()
L = L.index.to_list()
R = R.index.to_list()
V = V.index.to_list()
M = M.index.to_list()

# Variables
x_f = model.addVars(F, vtype=GRB.BINARY, name="x_f")
x_l = model.addVars(L, vtype=GRB.BINARY, name="x_l")
x_r = model.addVars(R, vtype=GRB.BINARY, name="x_r")
x_v = model.addVars(V, vtype=GRB.BINARY, name="x_v")
q_flp = model.addVars(F, L, P, vtype=GRB.CONTINUOUS, name="q_flp")
q_lrp = model.addVars(L, R, P, vtype=GRB.CONTINUOUS, name="q_lrp")
q_rmp = model.addVars(R, M, P, vtype=GRB.CONTINUOUS, name="q_rmp")
q_mvp = model.addVars(M, V, P, vtype=GRB.CONTINUOUS, name="q_mvp")
q_rvp = model.addVars(R, V, P, vtype=GRB.CONTINUOUS, name="q_rvp")
q_vlp = model.addVars(V, L, P, vtype=GRB.CONTINUOUS, name="q_vlp")
q_vfp = model.addVars(V, F, P, vtype=GRB.CONTINUOUS, name="q_vfp")

# Objective
transportation = quicksum(tc * q_flp[f, l, p] * wru * distance_matrix.loc[f, l] for f in F for l in L for p in P) + quicksum(tc * q_lrp[l, r, p] * wru * distance_matrix.loc[l, r] for l in L for r in R for p in P) 
transportation += quicksum(tc * q_rmp[r, m, p] * wrpd * distance_matrix.loc[r, m] for r in R for m in M for p in P) 
transportation += quicksum(tc * q_mvp[m, v, p] * weolpd * distance_matrix.loc[m, v] for m in M for v in V for p in P) 
transportation += quicksum(tc * q_rvp[r, v, p] * wrtp * distance_matrix.loc[r, v] for r in R for v in V for p in P) 
transportation += quicksum(tc * q_vlp[v, l, p] * wru * distance_matrix.loc[v, l] for v in V for l in L for p in P) 
transportation += quicksum(tc * q_vfp[v, f, p] * wrp * distance_matrix.loc[v, f] for v in V for f in F for p in P)


installation = quicksum(installation_cost[(installation_cost.Type=="Factory") & (installation_cost.Location==f)]["Installation Cost"].values[0] * x_f[f] for f in F) 
installation += quicksum(installation_cost[(installation_cost.Type=="Logistics Node") & (installation_cost.Location==l)]["Installation Cost"].values[0] * x_l[l] for l in L) 
installation += quicksum(installation_cost[(installation_cost.Type=="Retrofit Center") & (installation_cost.Location==r)]["Installation Cost"].values[0] * x_r[r] for r in R) 
installation += quicksum(installation_cost[(installation_cost.Type=="Recovery Center") & (installation_cost.Location==v)]["Installation Cost"].values[0] * x_v[v] for v in V)

model.setObjective(transportation + installation, GRB.MINIMIZE)

# Constraints
# Demand fulfillment constraints
for m_seg in M:
    for p in P:
        model.addConstr(quicksum(q_rmp[r, m_seg, p] for r in R) == dm_mp.loc[m_seg, p])
        model.addConstr(quicksum(q_mvp[m_seg, v, p] for v in V) == dmeol_mp.loc[m_seg, p])

# Facility operation constraints
for f in F:
    for p in P:
        for l in L:
            model.addConstr(q_flp[f, l, p] <= big_M * x_f[f])
            model.addConstr(q_flp[f, l, p] <= big_M * x_l[l])
for l in L:
    for p in P:
        for r in R:
            model.addConstr(q_lrp[l, r, p] <= big_M * x_l[l])
            model.addConstr(q_lrp[l, r, p] <= big_M * x_r[r])
for r in R:
    for p in P:
        for m in M:
            model.addConstr(q_rmp[r, m, p] <= big_M * x_r[r])
for r in R:
    for p in P:
        for v in V:
            model.addConstr(q_rvp[r, v, p] <= big_M * x_r[r])
            model.addConstr(q_rvp[r, v, p] <= big_M * x_v[v])
for v in V:
    for p in P:
        for l in L:
            model.addConstr(q_vlp[v, l, p] <= big_M * x_v[v])
            model.addConstr(q_vlp[v, l, p] <= big_M * x_l[l])
for v in V:
    for p in P:
        for f in F:
            model.addConstr(q_vfp[v, f, p] <= big_M * x_v[v])
            model.addConstr(q_vfp[v, f, p] <= big_M * x_f[f])
for m in M:
    for p in P:
        for v in V:
            model.addConstr(q_mvp[m, v, p] <= big_M * x_v[v])

# Flow conservation constraints
for f in F:
    for p in P:
        model.addConstr(quicksum(q_flp[f, l, p] for l in L) >= quicksum(q_vfp[v, f, p] for v in V))
for l in L:
    for p in P:
        model.addConstr(quicksum(q_flp[f, l, p] for f in F) + quicksum(q_vlp[v, l, p] for v in V) == quicksum(q_lrp[l, r, p] for r in R))
for r in R:
    for p in P:
        model.addConstr(quicksum(q_lrp[l, r, p] for l in L) == quicksum(q_rmp[r, m, p] for m in M))
for r in R:
    for p in P:
        model.addConstr(quicksum(q_rmp[r, m, p] for m in M) == quicksum(q_rvp[r, v, p] for v in V))
for v in V:
    for p in P:
        model.addConstr(quicksum(q_vlp[v, l, p] for l in L) + quicksum(q_vfp[v, f, p] for f in F) <= quicksum(q_mvp[m, v, p] for m in M))

model.optimize()
df_result = pd.DataFrame(columns=["Type", "Value"])
if model.status == GRB.OPTIMAL:
    print("Optimal solution found with total cost", model.objVal)
    for v in model.getVars():
        if v.x > 0:
            list_v = [v.varName, v.x]
            df_result = df_result.append(pd.Series(list_v, index=df_result.columns), ignore_index=True)

# st.write(df_result)
df_result_x = df_result[df_result.Type.str.contains("x")]
df_result_q = df_result[df_result.Type.str.contains("q")]

df_result_x["Location"] = df_result_x["Type"].str.extract(r'(\d+)')
df_result_x["Type"] = df_result_x["Type"].str.split("_").str[1].str.split("[").str[0]



df["Open"] = 0

map_indice_to_type = {
    "f": "Factory",
    "l": "Logistics Node",
    "r": "Retrofit Center",
    "v": "Recovery Center"
}

df_result_x["Type"] = df_result_x["Type"].apply(lambda x: map_indice_to_type[x[0]])
# st.write(df_result_x)

for t in type_to_icon.keys():
    for i, row in df_result_x[df_result_x["Type"] == t].iterrows():
        location = row["Location"]
        type_to_assign = row["Type"]
        value_to_assign = row["Value"]
        # Find indices in df to update
        indices_to_update = df[(df["Type"]==type_to_assign)  & (df["COM"] == location)].index
        for index in indices_to_update:
            df.loc[index, "Open"] = value_to_assign

st.write(f'''
         - Total cost for this network design: {round(model.objVal)};
         - Number of open factories **{df_result_x[df_result_x["Type"] == "Factory"]["Value"].count()}** over {nb_f};
         - Number of open logistics nodes **{df_result_x[df_result_x["Type"] == "Logistics Node"]["Value"].count()}** over {nb_l};
         - Number of open retrofit centers **{df_result_x[df_result_x["Type"] == "Retrofit Center"]["Value"].count()}** over {nb_r};
         - Number of open recovery centers **{df_result_x[df_result_x["Type"] == "Recovery Center"]["Value"].count()}** over {nb_v};
         ''')

fig = go.Figure()

# Loop through each facility type and add a scattermapbox trace for each
for facility_type, group_df in df.groupby('Type'):
    for status, status_df in group_df.groupby('Open'):
        if status == 1:
            opacity = 1
        else:
            opacity = 0.3
        fig.add_trace(go.Scattermapbox(
                lat=status_df['lat'],
                lon=status_df['lon'],
                mode='markers',
                marker=go.scattermapbox.Marker(
                    size=10,
                    opacity=opacity,
                    symbol=type_to_icon[facility_type],
                    color=type_to_color[facility_type]
                ),
                name=f"{facility_type} {status}"
            ))


df_result_q["from"] = df_result_q["Type"].str.split(",").str[0].str[-5:]
df_result_q["to"] = df_result_q["Type"].str.split(",").str[1].str[:5]
df_result_q["Product"] = df_result_q["Type"].str.split(",").str[2].str.split("]").str[0]
df_result_q["Type"] = df_result_q["Type"].str.split("_").str[1].str.split("p").str[0]

df_result_q["from_name"] = df_result_q["from"].map(commune_10k["Commune"])
df_result_q["to_name"] = df_result_q["to"].map(commune_10k["Commune"])


st.subheader("Capacity of each facility")
st.write("**Factories**")
st.plotly_chart(px.bar(df_result_q[df_result_q["Type"]=="fl"].groupby("from_name")["Value"].sum()).update_xaxes(type='category'))
st.write("**Logistics Nodes**")
st.plotly_chart(px.bar(df_result_q[df_result_q["Type"]=="lr"].groupby("from_name")["Value"].sum()).update_xaxes(type='category'))
st.write("**Retrofit Centers**")
st.plotly_chart(px.bar(df_result_q[df_result_q["Type"]=="rm"].groupby("from_name")["Value"].sum()).update_xaxes(type='category'))
st.write("**Recovery Centers**")
st.plotly_chart(px.bar(df_result_q[df_result_q["Type"]=="mv"].groupby("to_name")["Value"].sum()).update_xaxes(type='category'))

st.subheader("Quality of Service")
st.write("**From retrofit center to market segment**")
df_result_q["Distance"] = df_result_q.apply(lambda x: distance_matrix.loc[x["from"], x["to"]], axis=1)
df_result_q_rm = df_result_q[df_result_q["Type"]=="rm"]
df_result_q_mv = df_result_q[df_result_q["Type"]=="mv"]
ref_dist_rm = st.number_input('Reference distance', value=50, min_value=0, max_value=1000, step=10, key="rm")
st.write(f'Percentage of volume exceeding the reference distance {round(df_result_q_rm[df_result_q_rm["Distance"] > ref_dist_rm]["Value"].sum()/df_result_q_rm["Value"].sum()*100, 2)}%')

scatter_chart_rm = px.scatter(df_result_q_rm, x="Distance", y="Value", color="Product")
scatter_chart_rm.add_bar(x=[ref_dist_rm], y=[df_result_q_rm["Value"].max()], name="Reference Distance")
st.plotly_chart(scatter_chart_rm)

st.write("**From market segment to recovery center**")
ref_dist_mv = st.number_input('Reference distance', value=50, min_value=0, max_value=1000, step=10, key="mv")
st.write(f'Percentage of volume exceeding the reference distance {round(df_result_q_mv[df_result_q_mv["Distance"] > ref_dist_mv]["Value"].sum()/df_result_q_mv["Value"].sum()*100, 2)}%')

scatter_chart_mv = px.scatter(df_result_q_mv, x="Distance", y="Value", color="Product")
scatter_chart_mv.add_bar(x=[ref_dist_mv], y=[df_result_q_mv["Value"].max()], name="Reference Distance")
st.plotly_chart(scatter_chart_mv)

def map_to_forward_reverse(type):
    if type in ["fl", "lr", "vm"]:
        return "forward"
    else:
        return "reverse"

df_result_q["Direction"] = df_result_q["Type"].apply(lambda x: map_to_forward_reverse(x))
# st.write(df_result_q)
value_max= df_result_q["Value"].max()
if st.button("Show Flow"):
    for flow in df_result_q.iterrows():
        from_location = flow[1]["from"]
        to_location = flow[1]["to"]
        product = flow[1]["Product"]
        direction = flow[1]["Direction"]
        volume = flow[1]["Value"]
        if direction == "forward":
            fig.add_trace(go.Scattermapbox(
                lat=[commune_10k.loc[from_location, "lat"], commune_10k.loc[to_location, "lat"]],
                lon=[commune_10k.loc[from_location, "lon"], commune_10k.loc[to_location, "lon"]],
                mode='lines',
                line=dict(width=5*volume/value_max, color="blue"),
                text=f"{product} from {from_location} to {to_location} with volume {volume}",
                name ="Forward Flow"
                        ))
        else:
            fig.add_trace(go.Scattermapbox(
                lat=[commune_10k.loc[to_location, "lat"], commune_10k.loc[from_location, "lat"]],
                lon=[commune_10k.loc[to_location, "lon"], commune_10k.loc[from_location, "lon"]],
                mode='lines',
                line=dict(width=5*volume/value_max, color="green"),
                text=f"{product} from {to_location} to {from_location} with volume {volume}",
                name ="Reverse Flow"
            ))

    # Update layout with Mapbox access token and style
    fig.update_layout(
        mapbox_style="open-street-map",  # or use "open-street-map" if you don't have a Mapbox token
        mapbox_zoom=3.5,
        mapbox_center={"lat": df['lat'].median(), "lon": df['lon'].median()})


    st.plotly_chart(fig)

st.sidebar.title("Table of Contents")
st.sidebar.markdown("""
- [Illustration](#illustration)
- [Model](#model)
- [Potential Facility Locations](#potential-locations)
- [Results](#results)
""")
    