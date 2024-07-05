import sys
sys.path.append("..")
from tools import *
create_toc(["Location", "Decision Variables", "Parameters", "Objective Function", "Constraints", "Implementation"])

st.header("Integrated Model")

'''
# Location
## Decision Variables
### Facility
*Openness*

- $x_{ft}$ indicates whether factory $f$ is operational in time period $t$ (1 open; 0 closed)
- $x_{lt}$ indicates whether logistics node $l$ is operational in time period $t$ (1 open; 0 closed)
- $x_{rt}$ indicates whether retrofit center $r$ is operational in time period $t$ (1 open; 0 closed)
- $x_{vt}$ indicates whether recovery center $v$ is operational in time period $t$ (1 open; 0 closed)

*Allocation*

$z_{jj't}, (j, j')\in J^2$ indicates whether the flow from $j$ to $j'$ is authorized 

### Flow
- **Retrofit Units (RU)**:  $q_{ii'pt}^{RU}$  represents the flow of retrofit units from node $i$ to node $i'$ for product model $p$ at time $t$.
- **Products to be Retrofitted (PR)**:  $q_{ii'pt}^{PR}$  represents the flow of Products to be Retrofitted (PR) from node $i$ to node $i'$ for product model $p$ at time $t$.
- **Retrofitted Products (RP)**: $q_{ii'pt}^{RP}$ represents the flow of Retrofitted Products (RP) from node $i$ to node $i'$ for product model $p$ at time $t$.
- **EoL Product (EoLP)**: $q_{ii'pt}^{EoLP}$ represents the flow of EoL Product (EoLP) from node $i$ to node $i'$ for product model $p$ at time $t$.
- **EoL Retrofit Units (EoLRU)**: $q_{ii'pt}^{EoLRU}$ represents the flow of EoL Retrofit Units (EoLRU) from node $i$ to node $i'$ for product model $p$ at time $t$. 
- **EoL Parts (EoLPa)**:  $q_{ii'pt}^{EoLPa}$ represents the flow of EoL Parts (EoLPa) from node $i$ to node $i'$ for product model $p$ at time $t$.
- **Transport Unit (TR)**: $q_{ii't}^{TR}$ represents the flow of transport units from node $i$ to node $i'$ for product model $p$ at time $t$.

### Lost Orders
- $lo^{PR}_{mpt}$ lost order for retrofit of vehicle model $p$ at market segment $m$ at time period $t$
- $lo^{EoLP}_{mpt}$ lost order for EoL treatement of vehicle model $p$ at market segment $m$ at time period $t$

### EoL Orders
$dm^{EoLP}_{mpt}$ EoL treatement demand of vehicle model $p$ at market segment $m$ at time period $t$ 

## Parameters
*Maximal Capacity:*

- $maxcapMN_{f}$ maximal capacity of manufacturing at factory $f$
- $maxcapRMN_{f}$ maximal capacity of remanufacturing at factory $f$
- $maxcapH_{l}$ maximal capacity of handling at logistics node $l$
- $maxcapR_{r}$ maximal capacity of retrofitting at retrofit center $r$
- $maxcapDP_{r}$ maximal capacity of disassembling EoL product to retrieve EoL retrofit units at retrofit center $r$
- $maxcapRF_{v}$ maximal capacity of refurbishing at recovery center $v$
- $maxcapDRU_{v}$ maximal capacity of disassembling retrofit unit at recovery center $v$

*Minimum Operating Level*
- $molMN_{f}$ minimum operating level of manufacturing at factory $f$
- $molRMN_{f}$ minimum operating level of remanufacturing at factory $f$
- $molH_{l}$ minimum operating level of handling at logistics node $l$
- $molR_{r}$ minimum operating level of retrofitting at retrofit center $r$
- $molDP_{r}$ minimum operating level of disassembling EoL product to retrieve EoL retrofit units at retrofit center $r$
- $molRF_{v}$ minimum operating level of refurbishing at recovery center $v$
- $molDRU_{v}$ minimum operating level of disassembling retrofit unit at recovery center $v$

*Weight*
- $w^{RU}$ retrofit unit weight in kg (RU, EoLRU)
- $w^{EoLPa}$ average weight of EoL Parts (EoLPa)

*Distance*
- $d_{ii'}$ distance in km from node $i$ to node $i'$ with $(i, i')\in I^2$
- $D_{max}$ the maximal distance between a market segment and a retrofit center that the market segment can be served by the retrofit center

*Activation Footprint*
- $af_{j}$ facility activation footprint for $j\in J$

*Unit Operation Footprint*
- $uofMN_{f}, uofRMN_{f}$ unit operation footprint for manufacturing, remanufacturing at factory $f$
- $uofH_{l}$ unit operation footprint for handling at logistics node $l$
- $uofR_{r}$ unit operation footprint for retrofit at retrofit center $r$
- $uofDP_{r}$ unit operation footprint for deassembling EoL product and retrieve EoL retrofit unit at retrofit center $r$
- $uofRF_{v}$ unit operation footprint for refurbishing EoL retrofit unit at recovery center $v$
- $uofDRU_{v}$ unit operation footprint for disassembling retrofit unit to retrive and refurbish EoL parts at recovery center $v$

*Demand*
- $dm_{mpt}$ retrofit demand of vehicle model $p$ at market segment $m$ at time period $t$
- $pb^{EoL}_{\\tau}$ the percentage of retrofitted vehicles at end-of-life after $\\tau$ years (modeled by the Weibull distribution)

*Transportation*
- $pl^{TR}$ payload capacity per transport unit in kg
- $tf^{TR}$ transportation footprint per transport unit per km
- $fr^{TR}$ average filling rate per transport unit
- $utf^{PR}$ average transportation footprint per km for a vehicle to be retrofitted
- $utf^{RP}$ average transportation footprint per km for a retrofitted vehicle

*Lost order footprint*
- $lof^{PR}_p$ lost order footprint for retrofit per product for model $p$ 
- $lof^{EoLP}_p$ lost order footprint for EoL treatement per product for model $p$

$Z$ a large enough number

## Objective Function
Minimisation

Supply Chain Carbon Footprint = Facility Activation Carbon Footprint + Operation Carbon Footprint + Transport Footprint + Lost Orders Carbon Footprint

Facility Activation Carbon Footprint = Activation Action \*  Activatation Carbon Footprint

Operation Carbon Footprint = Volume \* Unit Operation Footprint

Transport Carbon Footprint = Number of transport units \* Footprint per transport unit per km \* Distance

Lost Orders Carbon Footprint = Volume of lost orders \* lost order footprint per product

### Facility Activation Carbon Footprint
$\sum_{j \in J} x_{jt_{last}}\cdot af_{j}$

### Operation Carbon Footprint 
Factories (F) : manufacturing operation footprint + remanufacturing operation footprint
$\sum_{t \in T} \sum_{p \in P}\sum_{f \in F}\left(\sum_{v \in V} q_{vfpt}^{EoLPa} \cdot uofRMN_{f} + (\sum_{l \in L} q_{flpt}^{RU} - \sum_{v \in V}q_{vfpt}^{EoLPa}) \cdot uofMN_{f} \\right)$ 

Logistics Nodes (L): handling operation footprint
$\sum_{t \in T}\sum_{p \in P} \sum_{l \in L} \left( \sum_{r \in R} q_{lrpt}^{RU} \cdot uofH_{l}\\right)$

Retrofit Centers (R): retrofit operation footprint and EoL vehicles disassembling footprint 
$\sum_{t \in T}\sum_{p \in P} \sum_{r \in R}\left(\sum_{m \in M}q^{RP}_{rmpt}\cdot uofR_{r} + \sum_{v \in V}q_{rvpt}^{EoLRU}\cdot uofDP_{r}\\right)$

Recovery Center (V): refurbish footprint and EoL retrofit unit disassembling footprint 
$\sum_{t \in T} \sum_{p \in P} \sum_{v \in V}  \left(\sum_{l\in L}q_{vlpt}^{RU} \cdot uofRF_{v} + \sum_{f\in F}q_{vfpt}^{EoLPa} \cdot uofDRU_{v} \\right)$

### Transport Carbon Footprint (same as evaluation model)
The sum of footprint for transport unit flows, retrofitted vehicles flows, and flows of vehicles to be retrofitted
$\sum_{p\in P}\sum_{t\in T}\left(\sum_{(i,i')\in I^2 \&i\\neq i'}q_{ii't}^{TR} \cdot tf^{TR} \cdot d_{ii'} + \sum_{(m,r)\in M\\times R}q_{mrpt}^{PR} \cdot utf^{PR} \cdot d_{mr} + \sum_{(r,m)\in R\\times M}q_{rmpt}^{RP} \cdot utf^{RP} \cdot d_{rm} + \sum_{(m,r)\in M\\times R}q_{mrpt}^{EoLP} \cdot utf^{RP}\cdot d_{mr}\\right)$

### Carbon Footprint for lost orders (same as evaluation model)
$\sum_{t\in T}\sum_{p\in P}\sum_{m \in M}(lo_{mpt}^{PR} \cdot lof^{PR}_p + lo_{mpt}^{EoLP}\cdot lof_p^{EoLP})$

## Constraints
### Calculation of EoL recovery demand
$d^{EoLP}_{mp1} = 0 \quad \\forall m\in M, \\forall p\in P$

$d^{EoLP}_{mpt} = \sum_{\\tau\in[1,t-1]}(\sum_{r\in R}q_{mrp\\tau}^{PR}) \cdot pb_{t-\\tau} - \sum_{\\tau\in[1, t-1]}d_{mp\\tau}^{EoLP} \quad \\forall m\in M, \\forall p\in P, \\forall t\in [2,|T|]$

### Demand fulfillment for retrofit and EoL products (same as evaluation model)
$\sum_{r \in R} q^{PR}_{mrpt} = dm_{mpt} - lo_{mpt} \quad \\forall m \in M, \\forall p \in P, \\forall t \in T$

$\sum_{r \in R} q^{EoLP}_{mrpt} = dm^{EoLP}_{mpt} - lo^{EoLP}_{mpt} \quad \\forall m \in M, \\forall p \in P, \\forall t \in T$

### Calculation of transport unit flows (same as evaluation model)
$\sum_{p\in P}\left( (q_{ii'pt}^{RU}+q_{ii'pt}^{EoLRU})\cdot w^{RU} + q_{ii'pt}^{EoLPa}\cdot w^{EoLPa} \\right)\leq q^{TR}_{ii't}\cdot pl^{TR} \cdot fr^{TR} \quad \\forall t \in T, \\forall (i,i')\in I^2 \& i\\neq i'$

### Capacity and minimum operating level constraints 
Factory:

*Manufacturing* $x_{ft} \cdot molMN_{ft} \leq \sum_{p\in P}(\sum_{l\in L} q^{RU}_{flpt} - \sum_{v\in V} q^{EoLPa}_{vfpt}) \leq x_{ft} \cdot capMN_{ft} \quad \\forall f \in F, \\forall t \in T$

*Remanufacturing* $x_{ft} \cdot molRMN_{ft} \leq \sum_{p\in P}\sum_{v\in V } q^{EoLPa}_{vfpt} \leq x_{ft} \cdot capRMN_{ft} \quad \\forall f \in F, \\forall t \in T$

Logistics node:

*Handling* $x_{lt} \cdot molH_{lt} \leq \sum_{p\in P}\sum_{r\in R} q^{RU}_{lrpt} \leq x_{lt} \cdot capH_{lt} \quad \\forall l \in L, \\forall t \in T$

Retrofit center:

*Retrofit* $x_{rt} \cdot molR_{rt} \leq \sum_{p\in P}\sum_{m\in M} q^{RP}_{rmpt}\leq x_{rt}\cdot capR_{rt} \quad \\forall r \in R, \\forall t \in T$

*Disassemble EoL Product* $x_{rt} \cdot molDP_{rt} \leq \sum_{p\in P}\sum_{m\in M}q^{EoLP}_{mrpt} \leq x_{rt}\cdot capDP_{rt} \quad \\forall r \in R, \\forall t \in T$

Recovery center:

*Refurbish* $x_{vt}\cdot molRF_{vt} \leq \sum_{p\in P}\sum_{l\in L}q_{vlpt}^{RU}\leq x_{vt}\cdot capRF_{vt}\quad \\forall v \in V, \\forall t \in T$

*Disassemble Retrofit Unit to retrieve and refurbish EoL parts* $x_{vt}\cdot molDRU_{vt} \leq \sum_{p\in P}\sum_{f\in F}q_{vfpt}^{EoLPa}\leq x_{vt}\cdot capDRU_{vt}\quad \\forall v \in V, \\forall t \in T$

### Flow conservation constraints (same as evaluation model)
All refurbished parts are used in the manufacturing of new retrofit kits

$\sum_{l\in L} q^{RU}_{flpt} \geq \sum_{v\in V} q^{EoLPa}_{vfpt} \quad \\forall f \in F, \\forall p \in P, \\forall t \in T$

For a logistics node, retrofit units that are sending out to retrofit centers or other logistics node are from either factory or recovery center

$\sum_{f \in F} q^{RU}_{flpt} + \sum_{v \in V} q^{RU}_{vlpt}  = \sum_{r \in R} q^{RU}_{lrpt}  \quad \\forall l \in L, \\forall p \in P, \\forall t \in T$

For a retrofit center, the products can only be retrofitted if the retrofit units are acquired 

$\sum_{l \in L} q^{RU}_{lrpt} = \sum_{m \in M} q^{PR}_{mrpt} \quad \\forall r \in R, \\forall p \in P, \\forall t \in T$

For a retrofit center, products to be retrofitted are getting retrofit in the same period

$q^{RP}_{rmpt} = q^{PR}_{mrpt} \quad \\forall r \in R, \\forall m \in M, \\forall p \in P, \\forall t \in T$

For a retrofit center, EoL products being disassembled have EoL retrofit units to be sent to recovery center

$\sum_{m \in M} q^{EoLP}_{mrpt} = \sum_{v \in V}q^{EoLRU}_{rvpt}\quad \\forall r \in R, \\forall p \in P, \\forall t \in T$

For a recovery center, the volume of refurbished parts and refurbished retrofit units should not exceed the volume of End-of-life retrofit units that are taken back

$\sum_{l \in L}q^{RU}_{vlpt} + \sum_{f\in F}q^{EoLPa}_{vfpt}\leq \sum_{r\in R}q^{EoLRU}_{rvpt} \quad \\forall v \in V, \\forall p \in P, \\forall t \in T$

### Allocation constraints
Flow can only exist when the path is authorized

$q_{ii't}^{TR}\leq Z\cdot z_{jj't} \quad \\forall (j,j')\in J^2 \& j\\neq j',  \\forall t \in T$

$q_{mrpt}^{RP}\leq Z\cdot z_{mrt} \quad \\forall (m,r)\in (M\\times R),  \\forall t \in T$

$q_{mrpt}^{EoLP}\leq Z\cdot z_{mrt} \quad \\forall (m,r)\in (M\\times R),  \\forall t \in T$

The paths are authorized only if the sites are open at the two ends

$z_{ii't}\leq x_{it} + x_{i't} -1$

Each factory send parts to a designated logistics node

$\sum_{l\in L}z_{flt} = 1 \quad \\forall f\in F, \\forall t\in T$

$z_{flt} = z_{fl(t+1)} \quad \\forall f\in F, \\forall l\in L, \\forall t\in T\\backslash\{|T|\}$

Each retrofit center is served solely by a specific logistics node

$\sum_{l\in L}z_{lrt} = 1 \quad \\forall r\in R, \\forall t\in T$

$z_{lrt} = z_{lr(t+1)} \quad \\forall l\in L, \\forall r\in R, \\forall t\in T\\backslash\{|T|\}$

Each retrofit center can only interact with a designated recovery center

$\sum_{v\in V}z_{rvt} = 1 \quad \\forall r\in R, \\forall t\in T$

$z_{rvt} = z_{rv(t+1)} \quad \\forall r\in R, v\in V, \\forall t\in T\\backslash\{|T|\}$

Each factory receives items from a specific recovery center

$\sum_{v\in V}z_{vft} = 1 \quad \\forall f\in F, \\forall t\in T$

$z_{vft} = z_{vf(t+1)} \quad \\forall v\in V, f\in F, \\forall t\in T\\backslash\{|T|\}$

Each recovery center send items to a designated logistics node

$\sum_{l\in L}z_{vlt} = 1 \quad \\forall v\in V, \\forall t\in T$

$z_{vlt} = z_{vl(t+1)} \quad \\forall v\in V, \\forall l\in L, \\forall t\in T\\backslash\{|T|\}$

### Openning constraints

$x_{jt}\leq x_{j(t+1)} \quad \\forall t\in\{1,2, ..., max(T)-1\}, \\forall j \in J$

### Distance constraints
$z_{mrt}\leq \mathbf{1}_{\{d_{rm}\leq D_{max}\}} \quad \\forall r\in R, \\forall m\in M$

### Non-negativity constraints
$q_{ii'pt}^{RU}, q_{ii'pt}^{PR}, q_{ii'pt}^{RP}, q_{ii'pt}^{EoLP}, q_{ii'pt}^{EoLRU}, q_{ii'pt}^{EoLPa} q_{ii't}^{TR} \geq 0 \quad \\forall (i,i')\in I^2 \& i\\neq i', \\forall p\in P, \\forall t\in T$

$lo^{PR}_{mpt}, lo^{EoLP}_{mpt} \geq 0 \quad \\forall m\in M, \\forall p\in P, \\forall t\in T$

## Implementation
'''

with open('parameters.pkl', 'rb') as f:
    M, P, T, demand, pb_eol, dist, maxcapMN, maxcapRMN, maxcapH, maxcapR, maxcapDP, maxcapRF, maxcapDRU, molMN, molRMN, molH, molR, molDP, molRF, molDRU, uofMN, uofRMN, uofH, uofR, uofDP, uofRF, uofDRU, wRU, wEoLPa, pl_TR, tf_TR, fr_TR, utf_PR, utf_RP, lof_PR, lof_EoLP, af, Z, D_max = pickle.load(f).values()

# Create a Gurobi model
model = Model("IntegratedModel")

F = M.apply(lambda x: f"{x}F")
L = M.apply(lambda x: f"{x}L")
R = M.apply(lambda x: f"{x}R")
V = M.apply(lambda x: f"{x}V")
M = M.apply(lambda x: f"{x}M")
I = pd.concat([F, L, R, V, M])
J = pd.concat([F, L, R, V])

# Decision variables

xf = model.addVars(F, T, vtype=GRB.BINARY, name="xf")
xl = model.addVars(L, T, vtype=GRB.BINARY, name="xl")
xr = model.addVars(R, T, vtype=GRB.BINARY, name="xr")
xv = model.addVars(V, T, vtype=GRB.BINARY, name="xv")

# q_RU = model.addVars(I, I, P, T, name="qRU")
q_RU = model.addVars(F, L, P, T, name="qRU") 
q_RU.update(model.addVars(L, R, P, T, name="qRU"))
q_RU.update(model.addVars(V, L, P, T, name="qRU"))
q_PR = model.addVars(M, R, P, T, name="qPR")
q_RP = model.addVars(R, M, P, T, name="qRP")
q_EoLP = model.addVars(M, R, P, T, name="qEoLP")
q_EoLRU = model.addVars(R, V, P, T, name="qEoLRU")
q_EoLPa = model.addVars(V, F, P, T, name="qEoLPa")
q_TR = model.addVars(F, L, T, vtype=GRB.INTEGER, name="qTR")
q_TR.update(model.addVars(L, R, T, vtype=GRB.INTEGER ,name="qTR"))
q_TR.update(model.addVars(R, V, T, vtype=GRB.INTEGER ,name="qTR"))
q_TR.update(model.addVars(V, F, T, vtype=GRB.INTEGER ,name="qTR"))
q_TR.update(model.addVars(V, L, T, vtype=GRB.INTEGER ,name="qTR"))
z = model.addVars(q_TR.keys(), vtype=GRB.BINARY, name="z")
z.update(model.addVars(M, R, T, vtype=GRB.BINARY, name="z"))

lo_PR = model.addVars(M, P, T, name="loPR")
lo_EoLP = model.addVars(M, P, T, name="loEoLP")

dm_EoLP = model.addVars(M, P, T, name="dmEoLP")


# Objective function
# Facility Activation Carbon Footprint
facility_activation_carbon_footprint = quicksum(xf[f, T[-1]] * af["F"] for f in F) + quicksum(xl[l, T[-1]] * af["L"] for l in L) + quicksum(xr[r, T[-1]] * af["R"] for r in R) + quicksum(xv[v, T[-1]] * af["V"] for v in V)

# Operation Carbon Footprint
operation_cf_factory = quicksum(quicksum(quicksum(quicksum(q_EoLPa[v, f, p, t] * uofRMN for v in V) + (quicksum(q_RU[f, l, p, t] for l in L) - quicksum(q_EoLPa[v, f, p, t] for v in V)) * uofMN for f in F) for p in P) for t in T)

operation_cf_logistics = quicksum(quicksum(quicksum((quicksum(q_RU[l, r, p, t] for r in R)) * uofH for l in L) for p in P) for t in T)

operation_cf_retrofit = quicksum(quicksum(quicksum(quicksum(q_RP[r, m, p, t] for m in M) * uofR + quicksum(q_EoLRU[r, v, p, t] for v in V) * uofDP for r in R) for p in P) for t in T)

operation_cf_recovery = quicksum(quicksum(quicksum(quicksum(q_RU[v, l, p, t] * uofRF for l in L) + quicksum(q_EoLPa[v, f, p, t] * uofDRU for f in F) for v in V) for p in P) for t in T)

operation_carbon_footprint = operation_cf_factory + operation_cf_logistics + operation_cf_retrofit + operation_cf_recovery


# Transport Carbon Footprint
transport_cf_TR = quicksum(quicksum(q_TR[i,i1,t] * tf_TR * dist.loc[i[:-1], i1[:-1]] for i in F for i1 in L) + quicksum(q_TR[i,i1,t] * tf_TR * dist.loc[i[:-1], i1[:-1]] for i in L for i1 in R) + quicksum(q_TR[i,i1,t] * tf_TR * dist.loc[i[:-1], i1[:-1]] for i in R for i1 in V) + quicksum(q_TR[i,i1,t] * tf_TR * dist.loc[i[:-1], i1[:-1]] for i in V for i1 in F) + quicksum(q_TR[i,i1,t] * tf_TR * dist.loc[i[:-1], i1[:-1]] for i in V for i1 in L) for t in T)

transport_cf = transport_cf_TR + quicksum(quicksum(quicksum(q_PR[m, r, p, t]*utf_PR*dist.loc[m[:-1], r[:-1]] for m in M for r in R) + quicksum(q_RP[r, m, p, t]*utf_RP*dist.loc[r[:-1], m[:-1]] for r in R for m in M) + quicksum(q_EoLP[m, r, p, t]*utf_RP*dist.loc[m[:-1], r[:-1]] for m in M for r in R) for t in T) for p in P)

# Lost Orders Carbon Footprint
lost_orders_cf = quicksum(lo_PR[m, p, t]*lof_PR + lo_EoLP[m, p, t]*lof_EoLP for m in M for p in P for t in T)

objective = facility_activation_carbon_footprint + operation_carbon_footprint + transport_cf + lost_orders_cf
model.setObjective(objective, GRB.MINIMIZE)

# Constraints
# Calculation of EoL recovery demand
for m in M:
    for p in P:
        for t in T:
            if t == T[0]:
                model.addConstr(dm_EoLP[m, p, t] == 0, name="EoL_demand_calculation_t0")
            else:
                model.addConstr(dm_EoLP[m, p, t] == quicksum((quicksum(q_PR[m, r, p, tau] for r in R)) * pb_eol[t-tau] - dm_EoLP[m, p, tau] for tau in range(1, t)), name="EoL_demand_calculation")
                                
# Demand fulfillment for retrofit and EoL products
for m in M:
    for p in P:
        for t in T:
            model.addConstr(quicksum(q_PR[m, r, p, t] for r in R) == demand.loc[(m, p), t] - lo_PR[m, p, t], name="demand_fulfillment_PR")
            model.addConstr(quicksum(q_EoLP[m, r, p, t] for r in R)== dm_EoLP[m, p, t] - lo_EoLP[m, p, t], name="demand_fulfillment_EoLP")

# Calculation of transport unit flows
for t in T:
    for i in F:
        for i1 in L:
                model.addConstr(quicksum(q_RU[i, i1, p, t] *wRU for p in P) <= q_TR[i, i1, t]*pl_TR*fr_TR, name="transport_unit_flow")
    for i in L:
        for i1 in R:
                model.addConstr(quicksum(q_RU[i, i1, p, t] *wRU for p in P) <= q_TR[i, i1, t]*pl_TR*fr_TR, name="transport_unit_flow")
    for i in V:
        for i1 in L:
                model.addConstr(quicksum(q_RU[i, i1, p, t] *wRU for p in P) <= q_TR[i, i1, t]*pl_TR*fr_TR, name="transport_unit_flow")
    for i in R:
        for i1 in V:
                model.addConstr(quicksum(q_EoLRU[i, i1, p, t] *wRU for p in P) <= q_TR[i, i1, t]*pl_TR*fr_TR, name="transport_unit_flow")
    for i in V:
        for i1 in F:
                model.addConstr(quicksum(q_EoLPa[i, i1, p, t] *wEoLPa for p in P) <= q_TR[i, i1, t]*pl_TR*fr_TR, name="transport_unit_flow")
    

# Capacity constraints
# factory
for f in F:
    for t in T:
        model.addConstr(quicksum(quicksum(q_RU[f, l, p, t] for l in L) - quicksum(q_EoLPa[v, f, p, t] for v in V) for p in P) >= xf[f, t]*molMN, name="factory_capacity manufacturing mol")
        model.addConstr(quicksum(quicksum(q_RU[f, l, p, t] for l in L) - quicksum(q_EoLPa[v, f, p, t] for v in V) for p in P) <= xf[f, t]*maxcapMN, name="factory_capacity manufacturing cap")
        model.addConstr(quicksum(quicksum(q_EoLPa[v, f, p, t] for v in V) for p in P) >= xf[f, t]*molRMN, name="factory_capacity remanufacturing mol")
        model.addConstr(quicksum(quicksum(q_EoLPa[v, f, p, t] for v in V) for p in P) <= xf[f, t]*maxcapRMN, name="factory_capacity remanufacturing cap")

# logistics node
for l in L:
    for t in T:
        model.addConstr(quicksum(quicksum(q_RU[l, r, p, t] for r in R) for p in P) >= xl[l, t]*molH, name="logistics_node_capacity mol")
        model.addConstr(quicksum(quicksum(q_RU[l, r, p, t] for r in R) for p in P) <= xl[l, t]*maxcapH, name="logistics_node_capacity")

# retrofit center
for r in R:
    for t in T:
        model.addConstr(quicksum(quicksum(q_RP[r, m, p, t] for m in M) for p in P) >= xr[r, t]*molR, name="retrofit_center_capacity retrofit mol")
        model.addConstr(quicksum(quicksum(q_RP[r, m, p, t] for m in M) for p in P) <= xr[r, t]*maxcapR, name="retrofit_center_capacity retrofit cap")
        model.addConstr(quicksum(quicksum(q_EoLP[m, r, p, t] for m in M) for p in P) >= xr[r, t]*molDP, name="retrofit_center_capacity disassemble mol")
        model.addConstr(quicksum(quicksum(q_EoLP[m, r, p, t] for m in M) for p in P) <= xr[r, t]*maxcapDP, name="retrofit_center_capacity disassemble cap")

# recovery center
for v in V:
    for t in T:
        model.addConstr(quicksum(quicksum(q_RU[v, l, p, t] for l in L) for p in P) >= xv[v, t]*molRF, name="recovery_center_capacity refurbish mol")
        model.addConstr(quicksum(quicksum(q_RU[v, l, p, t] for l in L) for p in P) <= xv[v, t]*maxcapRF, name="recovery_center_capacity refurbish cap")
        model.addConstr(quicksum(quicksum(q_EoLPa[v, f, p, t] for f in F) for p in P) >= xv[v, t]*molDRU, name="recovery_center_capacity disassemble mol")
        model.addConstr(quicksum(quicksum(q_EoLPa[v, f, p, t] for f in F) for p in P) <= xv[v, t]*maxcapDRU, name="recovery_center_capacity disassemble cap")

# Flow conservation constraints
for f in F:
    for p in P:
        for t in T:
            model.addConstr(quicksum(q_RU[f, l, p, t] for l in L) - quicksum(q_EoLPa[v, f, p, t] for v in V) >= 0, name="flow_conservation_factory")

for l in L:
    for p in P:
        for t in T:
            model.addConstr(quicksum(q_RU[f, l, p, t] for f in F) + quicksum(q_RU[v, l, p, t] for v in V) == quicksum(q_RU[l, r, p, t] for r in R), name="flow_conservation_logistics")

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
            model.addConstr(quicksum(q_EoLP[m, r, p, t] for m in M) == quicksum(q_EoLRU[r, v, p, t] for v in V), name="flow_conservation_retrofit_eol")

for v in V:
    for p in P:
        for t in T:
            model.addConstr(quicksum(q_RU[v, l, p, t] for l in L) + quicksum(q_EoLPa[v, f, p, t] for f in F) <= quicksum(q_EoLRU[r, v, p, t] for r in R), name="flow_conservation_recovery")

# Allocation constraints
for t in T:
    for j in F:
        for j1 in L:
            model.addConstr(q_TR[j, j1, t] <= Z*z[j, j1, t], name="allocation_TR")
    for j in L:
        for j1 in R:
            model.addConstr(q_TR[j, j1, t] <= Z*z[j, j1, t], name="allocation_TR")
    for j in R:
        for j1 in V:
            model.addConstr(q_TR[j, j1, t] <= Z*z[j, j1, t], name="allocation_TR")
    for j in V:
        for j1 in F:
            model.addConstr(q_TR[j, j1, t] <= Z*z[j, j1, t], name="allocation_TR")
    for j in V:
        for j1 in L:
            model.addConstr(q_TR[j, j1, t] <= Z*z[j, j1, t], name="allocation_TR")


for t in T:
    for m in M:
        for r in R:
            model.addConstr(q_RP[r, m, p, t] <= Z*z[m, r, t], name="allocation_RP")
            model.addConstr(q_EoLP[m, r, p, t] <= Z*z[m, r, t], name="allocation_EoLP")

for i, i_prime, t in z.keys():
    if i in F.values:
        xi_t = xf[i, t]
    elif i in L.values:
        xi_t = xl[i, t]
    elif i in R.values:
        xi_t = xr[i, t]
    elif i in V.values:
        xi_t = xv[i, t]

    if i_prime in F.values:
        xi_prime_t = xf[i_prime, t]
    elif i_prime in L.values:
        xi_prime_t = xl[i_prime, t]
    elif i_prime in R.values:
        xi_prime_t = xr[i_prime, t]
    elif i_prime in V.values:
        xi_prime_t = xv[i_prime, t]

    # Add the constraint
    model.addConstr(z[i, i_prime, t] <= xi_t)
    model.addConstr(z[i, i_prime, t] <= xi_prime_t)

for f in F:
    for t in T:
        model.addConstr(quicksum(z[f, l, t] for l in L) <= 1, name="logistics_factory_allocation")
    for l in L:
        for t in T[:-1]:
            model.addConstr(z[f, l, t] == z[f, l, t+1], name="logistics_factory_allocation_time")

for r in R:
    for t in T:
        model.addConstr(quicksum(z[l, r, t] for l in L) <= 1, name="retrofit_logistics_allocation")
    for l in L:
        for t in T[:-1]:
            model.addConstr(z[l, r, t] == z[l, r, t+1], name="retrofit_logistics_allocation_time")

for r in R:
    for t in T:
        model.addConstr(quicksum(z[r, v, t] for v in V) <= 1, name="retrofit_recovery_allocation")
    for v in V:
        for t in T[:-1]:
            model.addConstr(z[r, v, t] == z[r, v, t+1], name="retrofit_recovery_allocation_time")

for f in F:
    for t in T:
        model.addConstr(quicksum(z[v, f, t] for v in V) <= 1, name="factory_recovery_allocation")
    for v in V:
        for t in T[:-1]:
            model.addConstr(z[v, f, t] == z[v, f, t+1], name="factory_recovery_allocation_time")

for v in V:
    for t in T:
        model.addConstr(quicksum(z[v, l, t] for l in L) <= 1, name="logistics_recovery_allocation")
    for v in V:
        for t in T[:-1]:
            model.addConstr(z[v, l, t] == z[v, l, t+1], name="logistics_recovery_allocation_time")


# Openning constraints
for t in T[:-1]:
    for f in F:
        model.addConstr(xf[f, t] <= xf[f, t+1], name="factory_openning")
    for l in L:
        model.addConstr(xl[l, t] <= xl[l, t+1], name="logistics_openning")
    for r in R:
        model.addConstr(xr[r, t] <= xr[r, t+1], name="retrofit_openning")
    for v in V:
        model.addConstr(xv[v, t] <= xv[v, t+1], name="recovery_openning")

# Distance constraints
for r in R:
    for m in M:
        if dist.loc[m[:-1], r[:-1]] <= D_max:
            for t in T:
                model.addConstr(z[m, r, t] <= 1, name="distance_constraint")
        else:
            for t in T:
                model.addConstr(z[m, r, t] == 0, name="distance_constraint")

if st.button("Optimize"):
    model.optimize()
st.write(demand)
df_result = pd.DataFrame(columns=["Type", "Value"])
if model.status == GRB.OPTIMAL:
    st.write("Optimal solution found with total cost", model.objVal)
    for v in model.getVars():
        if v.x > 0:
            list_v = [v.varName, v.x]
            new_row = pd.Series(list_v, index=df_result.columns)
            df_result = pd.concat([df_result, new_row.to_frame().T], ignore_index=True)
else:
    st.write("No solution found")
st.write(df_result)




