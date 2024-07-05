import sys
sys.path.append("..")
from tools import *
import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))


st.header("Evaluation Model")

'''
## Types of flows:
- RU: retrofit units -- model dependent
- PR: products to be retrofitted -- model dependent
- RP: retrofitted product -- model dependent
- EoLRU: EoL retrofit units -- model dependent
- EoLPa: EoL parts -- model dependent
- TR: Transport Unit

## Types of operations:
- Factory: 
    - Manufacture (**MN**): from parts to RU
    - Remanufacture (**RMN**): from EoLPa to RU
- Logistics Node:
    - Handling (**H**): from RU to RU
- Retrofit Center:
    - Retrofit (**R**): from PR and RU to retrofitted product
    - Disassemble EoL Product to retrieve EoL retrofit units (**DP**): from EoL product to EoLRU
- Recovery Center:
    - Refurbish (**RF**): from EoLRU to RU
    - Disassemble Retrofit Unit to retrive and refurbish EoL parts (**DRU**): from EoLRU to EoLPa
'''


'''
## Hypothesis

- All the entered materials will be handled in the same period and will leave the node in the same period 
- Refurbished retrofit units are considered the same as new

## Sets
- $M$ set of all market segments
- $R$ set of all potential retrofit centers
- $V$ set of all potential recovery centers
- $L$ set of all potential logistics nodes
- $F$ set of all potential factories
- $I$ set of all nodes in the network with $I=\{ M \cup R \cup V \cup L \cup F\}$
- $J$ set of all potential sites with $J=\{ R \cup V \cup L \cup F\}$
- $P$ set of all product models
- $T$ set of time periods

## Decision Variables
- **Retrofit Units (RU)**:  $q_{ii'pt}^{RU}$  represents the flow of retrofit units from node $i$ to node $i'$ for product model $p$ at time $t$.
- **Products to be Retrofitted (PR)**:  $q_{ii'pt}^{PR}$  represents the flow of Products to be Retrofitted (PR) from node $i$ to node $i'$ for product model $p$ at time $t$.
- **Retrofitted Products (RP)**: $q_{ii'pt}^{RP}$ represents the flow of Retrofitted Products (RP) from node $i$ to node $i'$ for product model $p$ at time $t$.
- **EoL Product (EoLP)**: $q_{ii'pt}^{EoLP}$ represents the flow of EoL Product (EoLP) from node $i$ to node $i'$ for product model $p$ at time $t$.
- **EoL Retrofit Units (EoLRU)**: $q_{ii'pt}^{EoLRU}$ represents the flow of EoL Retrofit Units (EoLRU) from node $i$ to node $i'$ for product model $p$ at time $t$. 
- **EoL Parts (EoLPa)**:  $q_{ii'pt}^{EoLPa}$ represents the flow of EoL Parts (EoLPa) from node $i$ to node $i'$ for product model $p$ at time $t$.
- **Transport Unit (TR)**: $q_{ii't}^{TR}$ represents the flow of transport units from node $i$ to node $i'$ for product model $p$ at time $t$.
- **Lost Orders**
    - $lo^{PR}_{mpt}$ lost order for retrofit of vehicle model $p$ at market segment $m$ at time period $t$
    - $lo^{EoLP}_{mpt}$ lost order for EoL treatement of vehicle model $p$ at market segment $m$ at time period $t$

## Parameters
### Network Decisions
*Openness*
- $x_{ft}$ indicates whether factory $f$ is operational in time period $t$ (1 open; 0 closed)
- $x_{lt}$ indicates whether logistics node $l$ is operational in time period $t$ (1 open; 0 closed)
- $x_{rt}$ indicates whether retrofit center $r$ is operational in time period $t$ (1 open; 0 closed)
- $x_{vt}$ indicates whether recovery center $v$ is operational in time period $t$ (1 open; 0 closed)

*Capacity:*
- $capMN_{ft}$ capacity of manufacturing at time period $t$ at factory $f$
- $capRMN_{ft}$ capacity of remanufacturing at time period $t$ at factory $f$
- $capH_{lt}$ capacity of handling at time period $t$ at logistics node $l$
- $capR_{rt}$ capacity of retrofitting at time period $t$ at retrofit center $r$
- $capDP_{rt}$ capacity of disassembling EoL product to retrieve EoL retrofit units at retrofit center $r$
- $capRF_{vt}$ capacity of refurbishing at time period $t$ at recovery center $v$
- $capDRU_{vt}$ capacity of disassembling retrofit unit at time period $t$ at recovery center $v$

#### Others
*Minimum Operating Level*
- $molMN_{ft}$ minimum operating level of manufacturing at time period $t$ at factory $f$
- $molRMN_{ft}$ minimum operating level of remanufacturing at time period $t$ at factory $f$
- $molH_{lt}$ minimum operating level of handling at time period $t$ at logistics node $l$
- $molR_{rt}$ minimum operating level of retrofitting at time period $t$ at retrofit center $r$
- $molDP_{rt}$ minimum operating level of disassembling EoL product to retrieve EoL retrofit units at retrofit center $r$
- $molRF_{vt}$ minimum operating level of refurbishing at time period $t$ at recovery center $v$
- $molDRU_{vt}$ minimum operating level of disassembling retrofit unit at time period $t$ at recovery center $v$

*Weight*
- $w^{RU}$ retrofit unit weight in kg (RU, EoLRU)
- $w^{EoLPa}$ average weight of EoL Parts (EoLPa)

*Distance*
- $d_{ii'}$ distance in km from node $i$ to node $i'$

*Unit operation footprint*
- $uofMN_{f}, uofRMN_{f}$ unit operation footprint for manufacturing, remanufacturing at factory $f$
- $uofH_{l}$ unit operation footprint for handling at logistics node $l$
- $uofR_{r}$ unit operation footprint for retrofit at retrofit center $r$
- $uofDP_{r}$ unit operation footprint for deassembling EoL product and retrieve EoL retrofit unit at retrofit center $r$
- $uofRF_{v}$ unit operation footprint for refurbishing EoL retrofit unit at recovery center $v$
- $uofDRU_{v}$ unit operation footprint for disassembling retrofit unit to retrive and refurbish EoL parts at recovery center $v$

*Demand*
- $dm_{mpt}$ retrofit demand of vehicle model $p$ at market segment $m$ at time period $t$
- $dmEoLP_{mpt}$ EoL treatement demand of vehicle model $p$ at market segment $m$ at time period $t$

*Transportation*
- $pl^{TR}$ payload capacity per transport unit in kg
- $tf^{TR}$ transportation footprint per transport unit per km
- $fr^{TR}$ average filling rate per transport unit
- $utf^{PR}$ average transportation footprint per km for a vehicle to be retrofitted
- $utf^{RP}$ average transportation footprint per km for a retrofitted vehicle

*Lost order footprint*
- $lof^{PR}_p$ lost order footprint for retrofit per product for model $p$ 
- $lof^{EoLP}_p$ lost order footprint for EoL treatement per product for model $p$

## Objective function: 
Minimisation
Supply Chain Carbon Footprint = Operation Carbon Footprint + Transport Carbon Footprint

Operation Carbon Footprint = Volume \* Unit Operation Footprint

Transport Carbon Footprint = Number of transport units \* Footprint per transport unit per km \* Distance

#### 1. Operation Carbon Footprint 
Factories (F) : manufacturing operation footprint + remanufacturing operation footprint

$\sum_{t \in T} \sum_{p \in P}\sum_{f \in F}\left(\sum_{v \in V} q_{vfpt}^{EoLPa} \cdot uofRMN_{f} + (\sum_{l \in L} q_{flpt}^{RU} - \sum_{v \in V}q_{vfpt}^{EoLPa}) \cdot uofMN_{f} \\right)$

Logistics Nodes (L): handling operation footprint
$\sum_{t \in T}\sum_{p \in P} \sum_{l \in L} \left( (\sum_{r \in R} q_{lrpt}^{RU} + \sum_{l'\in L-\{l\}} q_{ll'pt}^{RU})\cdot uofH_{l}\\right)$

Retrofit Centers (R): retrofit operation footprint and EoL vehicles disassembling footprint 
$\sum_{t \in T}\sum_{p \in P} \sum_{r \in R}\left(\sum_{m \in M}q^{RP}_{rmpt}\cdot uofR_{r} + \sum_{v \in V}q_{rvpt}^{EoLRU}\cdot uofDP_{r}\\right)$

Recovery Center (V): refurbish footprint and EoL retrofit unit disassembling footprint 
$\sum_{t \in T} \sum_{p \in P} \sum_{v \in V}  \left(\sum_{l\in L}q_{vlpt}^{RU} \cdot uofRF_{v} + \sum_{f\in F}q_{vfpt}^{EoLPa} \cdot uofDRU_{v} \\right)$


### 2. Transport Carbon Footprint 
The sum of footprint for transport unit flows, retrofitted vehicles flows, and flows of vehicles to be retrofitted
$\sum_{p\in P}\sum_{t\in T}\left(\sum_{(i,i')\in I^2 \&i\\neq i'}q_{ii't}^{TR} \cdot tf^{TR} \cdot d_{ii'} + \sum_{(m,r)\in M\\times R}q_{mrpt}^{PR} \cdot utf^{PR} \cdot d_{mr} + \sum_{(r,m)\in R\\times M}q_{rmpt}^{RP} \cdot utf^{RP} \cdot d_{rm} + \sum_{(m,r)\in M\\times R}q_{mrpt}^{EoLP} \cdot utf^{RP}\cdot d_{mr}\\right)$

### 3. Carbon Footprint for lost orders
$\sum_{t\in T}\left(\sum_{p\in P}(\sum_{m \in M}lo_{mpt}^{PR} \cdot lof^{PR}_p + \sum_{m\in M}lo_{mpt}^{EoLP}\cdot lof_p^{EoLP})\\right)$

## Constraints
### 1. Demand fulfillment for retrofit and EoL products:
$\sum_{r \in R} q^{PR}_{mrpt} = dm_{mpt} - lo_{mpt} \quad \\forall m \in M, \\forall p \in P, \\forall t \in T$

$\sum_{r \in R} q^{EoLP}_{mrpt} = dmEoLP_{mpt} - loEoLP_{mpt} \quad \\forall m \in M, \\forall p \in P, \\forall t \in T$

### 2. Calculation of transport unit flows:
$\sum_{p\in P}\left( (q_{ii'pt}^{RU}+q_{ii'pt}^{EoLRU})\cdot w^{RU} + q_{ii'pt}^{EoLPa}\cdot w^{EoLPa} \\right)\leq q^{TR}_{ii't}\cdot pl^{TR} \cdot fr^{TR} \quad \\forall t \in T, \\forall (i,i')\in I^2 \& i\\neq i'$

### 3. Capacity constraints:

Factory:

*Manufacturing* $x_{ft} \cdot molMN_{ft} \leq \sum_{p\in P}(\sum_{l\in L} q^{RU}_{flpt} - \sum_{v\in V} q^{EoLPa}_{vfpt}) \leq x_{ft} \cdot capMN_{ft} \quad \\forall f \in F, \\forall t \in T$\\
*Remanufacturing* $x_{ft} \cdot molRMN_{ft} \leq \sum_{p\in P}\sum_{v\in V } q^{EoLPa}_{vfpt} \leq x_{ft} \cdot capRMN_{ft} \quad \\forall f \in F, \\forall t \in T$

Logistics node:

*Handling* $x_{lt} \cdot molH_{lt} \leq \sum_{p\in P}(\sum_{r\in R} q^{RU}_{lrpt} + \sum_{l'\in L-\{l\} }q^{RU}_{ll'pt})\leq x_{lt} \cdot capH_{lt} \quad \\forall l \in L, \\forall t \in T$

Retrofit center:

*Retrofit* $x_{rt} \cdot molR_{rt} \leq \sum_{p\in P}\sum_{m\in M} q^{RP}_{rmpt}\leq x_{rt}\cdot capR_{rt} \quad \\forall r \in R, \\forall t \in T$

*Disassemble EoL Product* $x_{rt} \cdot molDP_{rt} \leq \sum_{p\in P}\sum_{m\in M}q^{EoLP}_{mrpt} \leq x_{rt}\cdot capDP_{rt} \quad \\forall r \in R, \\forall t \in T$

Recovery center:

*Refurbish* $x_{vt}\cdot molRF_{vt} \leq \sum_{p\in P}\sum_{l\in L}q_{vlpt}^{RU}\leq x_{vt}\cdot capRF_{vt}\quad \\forall v \in V, \\forall t \in T$

*Disassemble Retrofit Unit to retrieve and refurbish EoL parts* $x_{vt}\cdot molDRU_{vt} \leq \sum_{p\in P}\sum_{f\in F}q_{vfpt}^{EoLPa}\leq x_{vt}\cdot capDRU_{vt}\quad \\forall v \in V, \\forall t \in T$

### 4. Flow conservation constraints:

All refurbished parts are used in the manufacturing of new retrofit kits

$\sum_{l\in L} q^{RU}_{flpt} \geq \sum_{v\in V} q^{EoLPa}_{vfpt} \quad \\forall f \in F, \\forall p \in P, \\forall t \in T$

For a logistics node, retrofit units that are sending out to retrofit centers or other logistics node are from either factory or recovery center

$\sum_{f \in F} q^{RU}_{flpt} + \sum_{v \in V} q^{RU}_{vlpt} + \sum_{l'\in L-\{l\}} q^{RU}_{l'lpt} = \sum_{r \in R} q^{RU}_{lrpt} + \sum_{l'\in L-\{l\}} q^{RU}_{ll'pt} \quad \\forall l \in L, \\forall p \in P, \\forall t \in T$

For a retrofit center, the products can only be retrofitted if the retrofit units are acquired 

$\sum_{l \in L} q^{RU}_{lrpt} = \sum_{m \in M} q^{PR}_{mrpt} \quad \\forall r \in R, \\forall p \in P, \\forall t \in T$

For a retrofit center, products to be retrofitted are getting retrofit in the same period

$q^{RP}_{rmpt} = q^{PR}_{mrpt} \quad \\forall r \in R, \\forall m \in M, \\forall p \in P, \\forall t \in T$

For a retrofit center, EoL products being disassembled have EoL retrofit units to be sent to recovery center

$\sum_{m \in M} q^{EoLP}_{mrpt} = \sum_{v \in V}q^{EoLRU}_{rvpt}\quad \\forall r \in R, \\forall p \in P, \\forall t \in T$

For a recovery center, the volume of refurbished parts and refurbished retrofit units should not exceed the volume of End-of-life retrofit units that are taken back

$\sum_{l \in L}q^{RU}_{vlpt} + \sum_{f\in F}q^{EoLPa}_{vfpt}\leq \sum_{r\in R}q^{EoLRU}_{rvpt} \quad \\forall v \in V, \\forall p \in P, \\forall t \in T$

#### 5. Non-negativity constraints

$q_{ii'pt}^{RU}, q_{ii'pt}^{PR}, q_{ii'pt}^{RP}, q_{ii'pt}^{EoLP}, q_{ii'pt}^{EoLRU}, q_{ii'pt}^{EoLPa} q_{ii't}^{TR}, lo^{PR}_{mpt}, lo^{EoLP}_{mpt} \geq 0 \quad \\forall (i,i')\in I^2 \& i\\neq i', \\forall p\in P$

'''

create_toc()

'''
## Datasets
Locations of the facilities
'''

locations = pd.read_excel(r"C:\Users\zwu\OneDrive - IMT Mines Albi\Documents\Data\ModelOptim\TestDataset\Locations.xlsx", dtype={"CodeCommuneInsee": "str"})
st.write(locations.groupby("TypeFacility").count().iloc[:,0].rename("Count").sort_values(ascending=True))
st.write(locations)
if st.button("Show on map"):
    fig = px.scatter_mapbox(locations, 
                            lat="Latitude", 
                            lon="Longitude", 
                            hover_name="CodeCommuneInsee", 
                            hover_data=["TypeFacility"],
                            color="TypeFacility",
                            zoom=5,
                            height=600)

    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    # Display the plot
    st.plotly_chart(fig)

# unique_codes = locations["CodeCommuneInsee"].unique()
# distance_matrix = pd.DataFrame(index=unique_codes, columns=unique_codes)
# total = len(unique_codes)**2
# for i in unique_codes:
#     lati = locations[locations["CodeCommuneInsee"]==i]["Latitude"].values[0]
#     loni = locations[locations["CodeCommuneInsee"]==i]["Longitude"].values[0]
#     for j in unique_codes:
#         latj = locations[locations["CodeCommuneInsee"]==j]["Latitude"].values[0]
#         lonj = locations[locations["CodeCommuneInsee"]==j]["Longitude"].values[0]
#         distance_matrix.loc[i, j] = haversine(lati, loni, latj, lonj)

# st.dataframe(distance_matrix)
# distance_matrix.to_csv(r"C:\Users\zwu\OneDrive - IMT Mines Albi\Documents\Data\ModelOptim\TestDataset\distance_matrix.csv")

distance_matrix = pd.read_csv(r"C:\Users\zwu\OneDrive - IMT Mines Albi\Documents\Data\ModelOptim\TestDataset\distance_matrix.csv", dtype={"Unnamed: 0": "str"}).set_index("Unnamed: 0")
st.write(distance_matrix)

M = locations[locations["TypeFacility"]=="Market Segment"]["CodeCommuneInsee"].values
R = locations[locations["TypeFacility"]=="Retrofit Center"]["CodeCommuneInsee"].values
V = locations[locations["TypeFacility"]=="Recovery Center"]["CodeCommuneInsee"].values
L = locations[locations["TypeFacility"]=="Logistics Node"]["CodeCommuneInsee"].values
F = locations[locations["TypeFacility"]=="Factory"]["CodeCommuneInsee"].values
I = locations["CodeCommuneInsee"].values
P = ["Fiat 500", "Renault Clio"]
T = [1, 2, 3]


np.random.seed(0)
# x_ft = pd.DataFrame(np.random.randint(2, size=(len(F),len(T))),index=F, columns=T)
# x_lt = pd.DataFrame(np.random.randint(2, size=(len(L),len(T))),index=L, columns=T)
# x_rt = pd.DataFrame(np.random.randint(2, size=(len(R),len(T))),index=R, columns=T)
# x_vt = pd.DataFrame(np.random.randint(2, size=(len(V),len(T))),index=V, columns=T)
x_ft = pd.DataFrame(np.ones((len(F),len(T)), dtype=int),index=F, columns=T)
x_lt = pd.DataFrame(np.ones((len(L),len(T))),index=L, columns=T)
x_rt = pd.DataFrame(np.ones((len(R),len(T))),index=R, columns=T)
x_vt = pd.DataFrame(np.ones((len(V),len(T))),index=V, columns=T)

capMN_ft = pd.DataFrame(np.random.randint(1000, 5000, size=(len(F),len(T)), dtype=int),index=F, columns=T)
capRMN_ft = pd.DataFrame(np.random.randint(200, 1000, size=(len(F),len(T)), dtype=int),index=F, columns=T)
capH_lt = pd.DataFrame(np.random.randint(1000, 5000, size=(len(L),len(T)), dtype=int),index=L, columns=T)
capR_rt = pd.DataFrame(np.random.randint(1000, 3000, size=(len(R),len(T)), dtype=int),index=R, columns=T)
capDP_rt = pd.DataFrame(np.random.randint(600, 2000, size=(len(R),len(T)), dtype=int),index=R, columns=T)
capRF_vt = pd.DataFrame(np.random.randint(1000, 5000, size=(len(V),len(T)), dtype=int),index=V, columns=T)
capDRU_vt = pd.DataFrame(np.random.randint(1000, 5000, size=(len(V),len(T)), dtype=int),index=V, columns=T)

molMN_ft = pd.DataFrame(np.random.randint(50, 100, size=(len(F),len(T)), dtype=int),index=F, columns=T)
molRMN_ft = pd.DataFrame(np.random.randint(10, 20, size=(len(F),len(T)), dtype=int),index=F, columns=T)
molH_lt = pd.DataFrame(np.random.randint(50, 100, size=(len(L),len(T)), dtype=int),index=L, columns=T)
molR_rt = pd.DataFrame(np.random.randint(5, 10, size=(len(R),len(T)), dtype=int),index=R, columns=T)
molDP_rt = pd.DataFrame(np.random.randint(3, 6, size=(len(R),len(T)), dtype=int),index=R, columns=T)
molRF_vt = pd.DataFrame(np.random.randint(5, 10, size=(len(V),len(T)), dtype=int),index=V, columns=T)
molDRU_vt = pd.DataFrame(np.random.randint(50, 100, size=(len(V),len(T)), dtype=int),index=V, columns=T)


wRU = 180
wEoLPa = 70

uofMN_f = pd.DataFrame(np.random.randint(1, 5, size=len(F), dtype=int),index=F)
uofRMN_f = pd.DataFrame(np.random.randint(1, 5, size=len(F), dtype=int),index=F)
uofH_l = pd.DataFrame(np.random.randint(1, 5, size=len(L), dtype=int),index=L)
uofR_r = pd.DataFrame(np.random.randint(1, 5, size=len(R), dtype=int),index=R)
uofDP_r = pd.DataFrame(np.random.randint(1, 5, size=len(R), dtype=int),index=R)
uofRF_v = pd.DataFrame(np.random.randint(1, 5, size=len(V), dtype=int),index=V)
uofDRU_v = pd.DataFrame(np.random.randint(1, 5, size=len(V), dtype=int),index=V)


grid1, grid2 = np.meshgrid(M,P)
index = pd.MultiIndex.from_tuples(list(zip(grid1.flatten(), grid2.flatten())))
dm_mpt = pd.DataFrame(np.random.randint(100, 1000, size=(len(M)*len(P),len(T))),index=index, columns=T)
dmEoLP_mpt = pd.DataFrame(np.random.randint(10, 100, size=(len(M)*len(P),len(T))),index=index, columns=T)

pl_TR = 1000
tf_TR = 0.1
fr_TR = 0.8
utf_PR = 0.2
utf_RP = 0.1
lofPR_p = 500
lofEoLP_p = 100

# Create a model
model = Model("Evaluation")

# decision variables
q_RU = model.addVars(I, I, P, T, name="q_RU") 
q_PR = model.addVars(I, I, P, T, name="q_PR")
q_RP = model.addVars(I, I, P, T, name="q_RP")
q_EoLP = model.addVars(I, I, P, T, name="q_EoLP")
q_EoLRU = model.addVars(I, I, P, T, name="q_EoLRU")
q_EoLPa = model.addVars(I, I, P, T, name="q_EoLPa")
q_TR = model.addVars(I, I, T, name="q_TR")
lo_PR = model.addVars(M, P, T, name="lo_PR")
lo_EoLP = model.addVars(M, P, T, name="lo_EoLP")
# objective function
operation_cf_factory = quicksum(quicksum(quicksum(quicksum(q_EoLPa[v, f, p, t] * uofRMN_f.loc[f,0] for v in V) + (quicksum(q_RU[f, l, p, t] for l in L) - quicksum(q_EoLPa[v, f, p, t] for v in V)) * uofMN_f.loc[f,0] for f in F) for p in P) for t in T)

operation_cf_logistics = quicksum(quicksum(quicksum((quicksum(q_RU[l, r, p, t] for r in R) + quicksum(q_RU[l, l1, p, t] for l1 in [l1 for l1 in L if l != l1])) * uofH_l.loc[l,0] for l in L) for p in P) for t in T)

operation_cf_retrofit = quicksum(quicksum(quicksum(quicksum(q_RP[r, m, p, t] * uofR_r.loc[r,0] for m in M) + quicksum(q_EoLRU[r, v, p, t] * uofDP_r.loc[r,0] for v in V) for r in R) for p in P) for t in T)

operation_cf_recovery = quicksum(quicksum(quicksum(quicksum(q_RU[v, l, p, t] * uofRF_v.loc[v,0] for l in L) + quicksum(q_EoLPa[v, f, p, t] * uofDRU_v.loc[v,0] for f in F) for v in V) for p in P) for t in T)

transport_cf = quicksum(quicksum(quicksum(q_TR[i, i1, t]*tf_TR*distance_matrix.loc[i, i1] for i in I for i1 in I if i != i1) + quicksum(q_PR[m, r, p, t]*utf_PR*distance_matrix.loc[m, r] for m in M for r in R) + quicksum(q_RP[r, m, p, t]*utf_RP*distance_matrix.loc[r, m] for r in R for m in M) + quicksum(q_EoLP[m, r, p, t]*utf_RP*distance_matrix.loc[m, r] for m in M for r in R) for t in T) for p in P)

lost_orders_cf = quicksum(lo_PR[m, p, t]*lofPR_p + lo_EoLP[m, p, t]*lofEoLP_p for m in M for p in P for t in T)

objfunc = operation_cf_factory + operation_cf_logistics + operation_cf_retrofit + operation_cf_recovery + transport_cf + lost_orders_cf

model.setObjective(objfunc, GRB.MINIMIZE)
# model.setObjective(lost_orders_cf, GRB.MINIMIZE)


# Demand fulfillment
for m in M:
    for p in P:
        for t in T:
            model.addConstr(quicksum(q_PR[m, r, p, t] for r in R) == dm_mpt.loc[(m, p), t] - lo_PR[m, p, t], name="demand_fulfillment_PR")
            model.addConstr(quicksum(q_EoLP[m, r, p, t] for r in R)== dmEoLP_mpt.loc[(m, p), t] - lo_EoLP[m, p, t], name="demand_fulfillment_EoLP")

# Calculation of transport unit flows
for t in T:
    for i in I:
        for i1 in I:
            if i != i1:
                model.addConstr(quicksum((q_RU[i, i1, p, t] + q_EoLRU[i, i1, p, t])*wRU + q_EoLPa[i, i1, p, t]*wEoLPa for p in P) <= q_TR[i, i1, t]*pl_TR*fr_TR, name="transport_unit_flow")

# Capacity constraints
# factory
for f in F:
    for t in T:
        model.addConstr(quicksum(quicksum(q_RU[f, l, p, t] for l in L) - quicksum(q_EoLPa[v, f, p, t] for v in V) for p in P) >= x_ft.loc[f, t]*molMN_ft.loc[f, t], name="factory_capacity")
        model.addConstr(quicksum(quicksum(q_RU[f, l, p, t] for l in L) - quicksum(q_EoLPa[v, f, p, t] for v in V) for p in P) <= x_ft.loc[f, t]*capMN_ft.loc[f, t], name="factory_capacity")
        model.addConstr(quicksum(quicksum(q_EoLPa[v, f, p, t] for v in V) for p in P) >= x_ft.loc[f, t]*molRMN_ft.loc[f, t], name="factory_capacity")
        model.addConstr(quicksum(quicksum(q_EoLPa[v, f, p, t] for v in V) for p in P) <= x_ft.loc[f, t]*capRMN_ft.loc[f, t], name="factory_capacity")

# logistics node
for l in L:
    for t in T:
        model.addConstr(quicksum((quicksum(q_RU[l, r, p, t] for r in R) + quicksum(q_RU[l, l1, p, t] for l1 in [l1 for l1 in L if l != l1])) for p in P) >= x_lt.loc[l, t]*molH_lt.loc[l, t], name="logistics_node_capacity")
        model.addConstr(quicksum((quicksum(q_RU[l, r, p, t] for r in R) + quicksum(q_RU[l, l1, p, t] for l1 in [l1 for l1 in L if l != l1])) for p in P) <= x_lt.loc[l, t]*capH_lt.loc[l, t], name="logistics_node_capacity")

# retrofit center
for r in R:
    for t in T:
        model.addConstr(quicksum(quicksum(q_RP[r, m, p, t] for m in M) for p in P) >= x_rt.loc[r, t]*molR_rt.loc[r, t], name="retrofit_center_capacity")
        model.addConstr(quicksum(quicksum(q_RP[r, m, p, t] for m in M) for p in P) <= x_rt.loc[r, t]*capR_rt.loc[r, t], name="retrofit_center_capacity")
        model.addConstr(quicksum(quicksum(q_EoLP[m, r, p, t] for m in M) for p in P) >= x_rt.loc[r, t]*molDP_rt.loc[r, t], name="retrofit_center_capacity")
        model.addConstr(quicksum(quicksum(q_EoLP[m, r, p, t] for m in M) for p in P) <= x_rt.loc[r, t]*capDP_rt.loc[r, t], name="retrofit_center_capacity")

# recovery center
for v in V:
    for t in T:
        model.addConstr(quicksum(quicksum(q_RU[v, l, p, t] for l in L) for p in P) >= x_vt.loc[v, t]*molRF_vt.loc[v, t], name="recovery_center_capacity")
        model.addConstr(quicksum(quicksum(q_RU[v, l, p, t] for l in L) for p in P) <= x_vt.loc[v, t]*capRF_vt.loc[v, t], name="recovery_center_capacity")
        model.addConstr(quicksum(quicksum(q_EoLPa[v, f, p, t] for f in F) for p in P) >= x_vt.loc[v, t]*molDRU_vt.loc[v, t], name="recovery_center_capacity")
        model.addConstr(quicksum(quicksum(q_EoLPa[v, f, p, t] for f in F) for p in P) <= x_vt.loc[v, t]*capDRU_vt.loc[v, t], name="recovery_center_capacity")

# flow conservation constraints
for f in F:
    for p in P:
        for t in T:
            model.addConstr(quicksum(q_RU[f, l, p, t] for l in L) >= quicksum(q_EoLPa[v, f, p, t] for v in V), name="flow_conservation_factory")

for l in L:
    for p in P:
        for t in T:
            model.addConstr(quicksum(q_RU[f, l, p, t] for f in F) + quicksum(q_RU[v, l, p, t] for v in V) + quicksum(q_RU[l, l1, p, t] for l1 in [l1 for l1 in L if l != l1]) == quicksum(q_RU[l, r, p, t] for r in R) + quicksum(q_RU[l, l1, p, t] for l1 in [l1 for l1 in L if l != l1]), name="flow_conservation_logistics")

for r in R:
    for p in P:
        for t in T:
            model.addConstr(quicksum(q_RU[l, r, p, t] for l in L) == quicksum(q_PR[m, r, p, t] for m in M), name="flow_conservation_retrofit")

for r in R:
    for m in M:
        for p in P:
            for t in T:
                model.addConstr(q_RP[r, m, p, t] == q_PR[m, r, p, t], name="flow_conservation_retrofit_product")

for r in R:
    for p in P:
        for t in T:
            model.addConstr(quicksum(q_EoLP[m, r, p, t] for m in M) == quicksum(q_EoLRU[r, v, p, t] for v in V), name="flow_conservation_retrofit_EoL")

for v in V:
    for p in P:
        for t in T:
            model.addConstr(quicksum(q_RU[v, l, p, t] for l in L) + quicksum(q_EoLPa[v, f, p, t] for f in F) <= quicksum(q_EoLRU[r, v, p, t] for r in R), name="flow_conservation_recovery")

model.optimize()
df_result = pd.DataFrame(columns=["Type", "Value"])
if model.status == GRB.OPTIMAL:
    st.write("Optimal solution found with total cost", model.objVal)
    for v in model.getVars():
        if v.x > 0:
            list_v = [v.varName, v.x]
            df_result = df_result.append(pd.Series(list_v, index=df_result.columns), ignore_index=True)
else:
    st.write("No solution found")
st.write(df_result)
st.write("Objective value: ", model.objVal)
# st.write(dm_mpt)
if model.status == GRB.INFEASIBLE:
    st.write("Model is infeasible")
    model.computeIIS()
    model.write("model.ilp")
    for c in model.getConstrs():
        if c.IISConstr:
            st.write(c.constrName)
                        