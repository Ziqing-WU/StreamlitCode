import sys
sys.path.append("..")
from tools import *
st.header("Hyperconnected Model")
'''
# Implementation
'''

with open('parameters_hyper.pkl', 'rb') as f:
    M, P, T, demand, pb_eol, dist, maxcapMN, maxcapRMN, maxcapH, maxcapR, maxcapDP, maxcapRF, maxcapDRU, molMN, molRMN, molH, molR, molDP, molRF, molDRU, uofMN, uofRMN, uofH, uofR, uofDP, uofRF, uofDRU, wRU, wEoLPa, pl_TR, tf_TR, fr_TR, utf_PR, utf_RP, lof_PR, lof_EoLP, af, of, Z, D_max = pickle.load(f).values()

model = Model("Together")

F = M.apply(lambda x: f"{x}F")
L = M.apply(lambda x: f"{x}L")
R = M.apply(lambda x: f"{x}R")
V = M.apply(lambda x: f"{x}V")
M = M.apply(lambda x: f"{x}M")
I = pd.concat([F, L, R, V, M])
J = pd.concat([F, L, R, V])

xf = model.addVars(F, T, vtype=GRB.BINARY, name="xf")
xl = model.addVars(L, T, vtype=GRB.BINARY, name="xl")
xr = model.addVars(R, T, vtype=GRB.BINARY, name="xr")
xv = model.addVars(V, T, vtype=GRB.BINARY, name="xv")

q_RU = model.addVars(F, L, P, T, name="qRU") 
q_RU.update(model.addVars(L, R, P, T, name="qRU"))
q_RU.update(model.addVars(V, L, P, T, name="qRU"))
q_RU.update(model.addVars(L, L, P, T, name="qRU"))
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
q_TR.update(model.addVars(L, L, T, vtype=GRB.INTEGER ,name="qTR"))
z = model.addVars(q_TR.keys(), vtype=GRB.BINARY, name="z")
z.update(model.addVars(M, R, T, vtype=GRB.BINARY, name="z"))

lo_PR = model.addVars(M, P, T, name="loPR")
lo_EoLP = model.addVars(M, P, T, name="loEoLP")

dm_EoLP = model.addVars(M, P, T, name="dmEoLP")

# Objective function
# Facility Activation Carbon Footprint
facility_activation_carbon_footprint = quicksum(xf[f,t] * of["F"] for f in F for t in T) + quicksum(xv[v, t] * of["V"] for v in V for t in T) + quicksum(xl[l, t] * of["L"] for l in L for t in T) + quicksum(xr[r, t] * of["R"] for r in R for t in T)

# Operation Carbon Footprint
operation_cf_factory = quicksum(quicksum(quicksum(quicksum(q_EoLPa[v, f, p, t] * uofRMN for v in V) + (quicksum(q_RU[f, l, p, t] for l in L) - quicksum(q_EoLPa[v, f, p, t] for v in V)) * uofMN for f in F) for p in P) for t in T)

operation_cf_logistics = quicksum(quicksum(quicksum((quicksum(q_RU[l, r, p, t] for r in R) + quicksum(q_RU[l, l1, p, t] for l1 in L if l != l1)) * uofH for l in L) for p in P) for t in T)

operation_cf_retrofit = quicksum(quicksum(quicksum(quicksum(q_RP[r, m, p, t] for m in M) * uofR + quicksum(q_EoLRU[r, v, p, t] for v in V) * uofDP for r in R) for p in P) for t in T)

operation_cf_recovery = quicksum(quicksum(quicksum(quicksum(q_RU[v, l, p, t] * uofRF for l in L) + quicksum(q_EoLPa[v, f, p, t] * uofDRU for f in F) for v in V) for p in P) for t in T)

operation_carbon_footprint = operation_cf_factory + operation_cf_logistics + operation_cf_retrofit + operation_cf_recovery

# Transport Carbon Footprint
transport_cf_TR = quicksum(quicksum(q_TR[i,i1,t] * tf_TR * dist.loc[i[:-1], i1[:-1]] for i in F for i1 in L) + quicksum(q_TR[i,i1,t] * tf_TR * dist.loc[i[:-1], i1[:-1]] for i in L for i1 in R) + quicksum(q_TR[i,i1,t] * tf_TR * dist.loc[i[:-1], i1[:-1]] for i in R for i1 in V) + quicksum(q_TR[i,i1,t] * tf_TR * dist.loc[i[:-1], i1[:-1]] for i in V for i1 in F) + quicksum(q_TR[i,i1,t] * tf_TR * dist.loc[i[:-1], i1[:-1]] for i in V for i1 in L) + quicksum(q_TR[i,i1,t] * tf_TR * dist.loc[i[:-1], i1[:-1]] for i in L for i1 in L if i != i1) for t in T)

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
            model.addConstr(quicksum(q_EoLP[m, r, p, t] for r in R) == dm_EoLP[m, p, t] - lo_EoLP[m, p, t], name="demand_fulfillment_EoLP")

# Calculation of transport unit flows
index_pairs = [
    (F, L),
    (L, R),
    (V, L),
    (L, L)
]

# Iterate over the time periods
for t in T:
    # Iterate over the defined index pairs
    for (source, target) in index_pairs:
        # Loop over all index pairs (i, i1) from source and target
        for i, i1 in product(source, target):
            if i != i1:
            # Add constraint to the model
                model.addConstr(
                    quicksum(q_RU[i, i1, p, t] * wRU for p in P) <= q_TR[i, i1, t] * pl_TR * fr_TR,
                    name="transport_unit_flow"
                )
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
        model.addConstr(quicksum((quicksum(q_RU[l, r, p, t] for r in R) + quicksum(q_RU[l, l1, p, t] for l1 in L if l != l1)) for p in P) >= xl[l, t]*molH, name="logistics_node_capacity")
        model.addConstr(quicksum((quicksum(q_RU[l, r, p, t] for r in R) + quicksum(q_RU[l, l1, p, t] for l1 in L if l != l1)) for p in P) <= xl[l, t]*maxcapH, name="logistics_node_capacity")

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
            model.addConstr(quicksum(q_RU[f, l, p, t] for f in F) + quicksum(q_RU[v, l, p, t] for v in V) + quicksum(q_RU[l, l1, p, t] for l1 in L if l!=l1) == quicksum(q_RU[l, r, p, t] for r in R) + quicksum(q_RU[l, l1, p, t] for l1 in L if l!=l1), name="flow_conservation_logistics")

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

for t in T:
    for m in M:
        for r in R:
            model.addConstr(q_RP[r, m, p, t] <= Z*z[m, r, t], name="allocation_RP")
            model.addConstr(q_EoLP[m, r, p, t] <= Z*z[m, r, t], name="allocation_EoLP")



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

