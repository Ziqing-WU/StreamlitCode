import sys
sys.path.append("..")
from tools import *

st.header("Integrated Model")

'''
# Implementation
'''

file_name = 'parameters.pkl'
with open(file_name, 'rb') as f:
    M, P, T, demand, pb_eol, dist, maxcapMN, maxcapRMN, maxcapH, maxcapR, maxcapDP, maxcapRF, maxcapDRU, molMN, molRMN, molH, molR, molDP, molRF, molDRU, uofMN, uofRMN, uofH, uofR, uofDP, uofRF, uofDRU, wRU, wEoLPa, pl_TR, tf_TR, fr_TR, utf_PR, utf_RP, lof_PR, lof_EoLP, af, of, Z, D_max = pickle.load(f).values()
current_time = datetime.now().strftime("%Y%m%d%H%M")
destination_path = "experimentations/integrated" + current_time + "/"
os.makedirs(destination_path, exist_ok=True)
os.rename(file_name, destination_path+file_name)

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

q_RU = model.addVars(F, L, P, T, name="qRU") 
q_RU.update(model.addVars(L, R, P, T, name="qRU"))
q_RU.update(model.addVars(V, L, P, T, name="qRU"))
q_PR = model.addVars(M, R, P, T, name="qPR")
q_RP = model.addVars(R, M, P, T, name="qRP")
q_EoLP = model.addVars(M, R, P, T, name="qEoLP")
q_EoLRU = model.addVars(R, V, P, T, name="qEoLRU")
q_EoLPa = model.addVars(V, F, P, T, name="qEoLPa")
# q_TR = model.addVars(F, L, T, vtype=GRB.INTEGER, name="qTR")
# q_TR.update(model.addVars(L, R, T, vtype=GRB.INTEGER ,name="qTR"))
# q_TR.update(model.addVars(R, V, T, vtype=GRB.INTEGER ,name="qTR"))
# q_TR.update(model.addVars(V, F, T, vtype=GRB.INTEGER ,name="qTR"))
# q_TR.update(model.addVars(V, L, T, vtype=GRB.INTEGER ,name="qTR"))
q_TR = model.addVars(F, L, T, name="qTR")
q_TR.update(model.addVars(L, R, T, name="qTR"))
q_TR.update(model.addVars(R, V, T, name="qTR"))
q_TR.update(model.addVars(V, F, T, name="qTR"))
q_TR.update(model.addVars(V, L, T, name="qTR"))
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

# objective = facility_activation_carbon_footprint + operation_carbon_footprint + transport_cf + lost_orders_cf
model.setObjective(facility_activation_carbon_footprint + operation_carbon_footprint + transport_cf + lost_orders_cf, GRB.MINIMIZE)

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
    (V, L)
]

# Iterate over the time periods
for t in T:
    # Iterate over the defined index pairs
    for (source, target) in index_pairs:
        # Loop over all index pairs (i, i1) from source and target
        for i, i1 in product(source, target):
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
        model.addConstr(quicksum(q_TR[r, v, t] for r in R) <= xv[v, t]*Z,  name="recovery_center openning")


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
            model.addConstr(quicksum(q_RU[v, l, p, t] for l in L) + quicksum(q_EoLPa[v, f, p, t] for f in F) == quicksum(q_EoLRU[r, v, p, t] for r in R), name="flow_conservation_recovery")

# Allocation constraints
index_pairs1 = [
    (F, L),
    (L, R),
    (R, V),
    (V, F),
    (V, L)
]

# Iterate over the time periods
for t in T:
    # Iterate over the defined index pairs
    for (source, target) in index_pairs1:
        # Loop over all index pairs (j, j1) from source and target
        for j, j1 in product(source, target):
            # Add constraint to the model
            model.addConstr(
                q_TR[j, j1, t] <= Z * z[j, j1, t],
                name="allocation_TR"
            )


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
    else:
        xi_t = 1

    if i_prime in F.values:
        xi_prime_t = xf[i_prime, t]
    elif i_prime in L.values:
        xi_prime_t = xl[i_prime, t]
    elif i_prime in R.values:
        xi_prime_t = xr[i_prime, t]
    elif i_prime in V.values:
        xi_prime_t = xv[i_prime, t]
    else:
        xi_prime_t = 1

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

set_param_optim(model, destination_path)
model.optimize()
st.write(demand)
if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT or model.status == GRB.INTERRUPTED:
    st.write("Optimal solution found with total cost", model.objVal)
    optim_obj = {"Model Status": model.status, "Total footprint": model.objVal, "Activation": facility_activation_carbon_footprint.getValue(), "Operation": operation_carbon_footprint.getValue(), "Transport": transport_cf.getValue(), "Lost orders": lost_orders_cf.getValue()}
    df_result = pd.DataFrame(optim_obj.items(), columns=["Type", "Value"])
    for v in model.getVars():
        if v.x > 0:
            list_v = [v.varName, v.x]
            new_row = pd.Series(list_v, index=df_result.columns)
            df_result = pd.concat([df_result, new_row.to_frame().T], ignore_index=True)
    st.write(df_result)
    df_result.to_csv(destination_path+"results.csv", index=True)
else:
    st.write("No solution found")





