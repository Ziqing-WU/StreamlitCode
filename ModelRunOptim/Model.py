from config import *



source_folder = "Parameters-Integrated"


for file_name in os.listdir(source_folder):

    with open(source_folder + "/" + file_name, "rb") as f:
        parameters = pickle.load(f)

    collab, M, R, V, L, F, P, T, demand, pb_EoL, dist, maxcapMN, maxcapRMN, maxcapH, maxcapR, maxcapDP, maxcapRF, maxcapDRU, molMN, molRMN, molH, molR, molDP, molRF, molDRU, uofMN, uofRMN, uofH, uofR, uofDP, uofRF, uofDRU, wRU, wEoLPa, pl_TR, tf_TR, fr_TR, utf_PR, utf_RP, lofPR, lofEoLP, af, Z, D_max = parameters.values()
    print(collab)
    current_time = datetime.now().strftime("%Y%m%d%H%M")
    destination_path = "experimentations/" + collab + "/" + current_time + "/"
    os.makedirs(destination_path, exist_ok=True) 
    os.rename(source_folder + "/" + file_name, destination_path+file_name)
    
    demand = demand.set_index("code_commune_titulaire", drop=True)
    
    model = Model(collab)
    F = F.apply(lambda x: f"{x}F")
    L = L.apply(lambda x: f"{x}L")
    R = R.apply(lambda x: f"{x}R")
    V = V.apply(lambda x: f"{x}V")
    M = pd.Series(M).apply(lambda x: f"{x}M")
    
    I = pd.concat([F, L, R, V, M])
    J = pd.concat([F, L, R, V])
    A_FL = list(product(F, L))  # (F x L)
    A_LR = list(product(L, R))  # (L x R)
    A_RV = list(product(R, V))  # (R x V)
    A_VF = list(product(V, F))  # (V x F)
    A_VL = list(product(V, L))  # (V x L)
    A_MR = list(product(M, R))  # (M x R)
    
    # Create the arcs (l, l') for l, l' in L where l != l'
    A_LL = [(l1, l2) for l1 in L for l2 in L if l1 != l2]
    
    # Combine all arcs to form A^TR
    A_TR = A_FL + A_LR + A_RV + A_VF + A_VL + A_LL
    A_RU = A_FL + A_LR + A_VL + A_LL
    A_all = A_FL + A_LR + A_MR + A_RV + A_VF + A_VL + A_LL
    
    # Decision variables
    x = model.addVars(J, T, vtype=GRB.BINARY, name="x")
    
    z = model.addVars(A_all, T, vtype=GRB.BINARY, name="z")
    
    qRU = model.addVars(A_RU, T, P, vtype=GRB.CONTINUOUS, name="qRU")
    qPR = model.addVars(M, R, T, P, vtype=GRB.CONTINUOUS, name="qPR")
    qRP = model.addVars(R, M, T, P, vtype=GRB.CONTINUOUS, name="qRP")
    
    qEoLP = model.addVars(M, R, T, P, vtype=GRB.CONTINUOUS, name="qEoLP")
    qEoLRU = model.addVars(R, V, T, P, vtype=GRB.CONTINUOUS, name="qEoLRU")
    qEoLPa = model.addVars(V, F, T, P, vtype=GRB.CONTINUOUS, name="qEoLPa")
    
    qTR = model.addVars(A_TR, T, vtype=GRB.INTEGER, name="qTR")
    loPR = model.addVars(M, T, P, vtype=GRB.CONTINUOUS, name="loPR")
    loEoLP = model.addVars(M, T, P, vtype=GRB.CONTINUOUS, name="loEoLP")
    
    dmEoLP = model.addVars(M, T, P, vtype=GRB.CONTINUOUS, name="dmEoLP")
    
    # Objective function
    CF_FA_F = quicksum(x[f, t] * af["F"] for f in F for t in T)
    CF_FA_L = quicksum(x[l, t] * af["L"] for l in L for t in T)
    CF_FA_R = quicksum(x[r, t] * af["R"] for r in R for t in T)
    CF_FA_V = quicksum(x[v, t] * af["V"] for v in V for t in T)
    CF_FA = CF_FA_F + CF_FA_L + CF_FA_R + CF_FA_V
    
    
    CF_OP_RMN = quicksum(qEoLPa[v, f, t, p] * uofRMN for v in V for f in F for p in P for t in T)
    CF_OP_MN = quicksum((quicksum(qRU[f, l, t, p] for l in L) - quicksum(qEoLPa[v, f, t, p] for v in V)) * uofMN for f in F for p in P for t in T)
    CF_OP_H = quicksum((quicksum(qRU[l, r, t, p] for r in R) + quicksum(qRU[l, l2, t, p] for l2 in [l1 for l1 in L if l1 != l])) * uofH for l in L for p in P for t in T)
    CF_OP_R = quicksum((quicksum(qRP[r, m, t, p] * uofR for m in M) + quicksum(qEoLRU[r, v, t, p] * uofDP for v in V)) for r in R for p in P for t in T)
    CF_OP_RF = quicksum((quicksum(qRU[v, l, t, p] * uofRF for l in L) + quicksum(qEoLPa[v, f, t, p] * uofDRU for f in F)) for v in V for p in P for t in T)
    CF_OP = CF_OP_RMN + CF_OP_MN + CF_OP_H + CF_OP_R + CF_OP_RF
    
    CF_TR_Trans = quicksum(qTR[i, i2, t] * tf_TR * dist.loc[i[:-1], i2[:-1]] for i, i2 in A_TR for t in T)
    CF_TR_V = quicksum(qPR[m, r, t, p] * utf_PR * dist.loc[m[:-1], r[:-1]] for m in M for r in R for p in P for t in T) + quicksum(qRP[r, m, t, p] * utf_RP * dist.loc[r[:-1], m[:-1]] for r in R for m in M for p in P for t in T) + quicksum(qEoLP[m, r, t, p] * utf_RP * dist.loc[m[:-1], r[:-1]] for m in M for r in R for p in P for t in T)
    CF_TR = CF_TR_Trans + CF_TR_V
    
    CF_LO_R = quicksum(loPR[m, t, p] * lofPR for m in M for p in P for t in T)
    CF_LO_EoLP = quicksum(loEoLP[m, t, p] * lofEoLP for m in M for p in P for t in T)
    CF_LO = CF_LO_R + CF_LO_EoLP
    
    model.setObjective(CF_FA + CF_OP + CF_TR + CF_LO, GRB.MINIMIZE)
    
    # Constraints
    dm_recovery_1 = model.addConstrs((dmEoLP[m, 1, p] == 0 for m in M for p in P), name="dm_recovery_1")
    dm_recovery_t = model.addConstrs(
        (
            dmEoLP[m, t, p] == quicksum(
                quicksum(qPR[m, r, tau, p] for r in R) * (pb_EoL[t - tau + 1] - pb_EoL[t - tau])
                for tau in range(1, t)
            )
            for m in M for p in P for t in range(2, len(T) + 1)
        ),
        name="dm_recovery_t"
    )
    
    demand_fulfillment_PR = model.addConstrs((quicksum(qPR[m, r, t, p] for r in R) == demand.loc[m[:-1], t] - loPR[m, t, p] for m in M for p in P for t in T), name="demand_fulfillment_PR")
    demand_fulfillment_EoLP = model.addConstrs((quicksum(qEoLP[m, r, t, p] for r in R) == dmEoLP[m, t, p] - loEoLP[m, t, p] for m in M for p in P for t in T), name="demand_fulfillment_EoLP")
    
    transport_unit_flows_RU = model.addConstrs((quicksum(qRU[i, i2, t, p] * wRU for p in P) <= qTR[i, i2, t] * pl_TR * fr_TR for i, i2 in A_RU for t in T), name="transport_unit_flows")
    transport_unit_flows_EoLRU = model.addConstrs((quicksum(qEoLRU[i, i2, t, p] * wRU for p in P) <= qTR[i, i2, t] * pl_TR * fr_TR for i, i2 in A_RV for t in T), name="transport_unit_flows_EoLRU")
    transport_unit_flows_EoLPa = model.addConstrs((quicksum(qEoLPa[i, i2, t, p] * wEoLPa for p in P) <= qTR[i, i2, t] * pl_TR * fr_TR for i, i2 in A_VF for t in T), name="transport_unit_flows_EoLPa")
    
    max_capa_MN = model.addConstrs(quicksum(quicksum(qRU[f, l, t, p] for l in L) - quicksum(qEoLPa[v, f, t, p] for v in V) for p in P) <= x[f, t] * maxcapMN for f in F for t in T)
    min_operating_MN = model.addConstrs(quicksum(quicksum(qRU[f, l, t, p] for l in L) - quicksum(qEoLPa[v, f, t, p] for v in V) for p in P) >= x[f, t] * molMN for f in F for t in T)
    max_capa_RMN = model.addConstrs(quicksum(qEoLPa[v, f, t, p] for v in V for p in P) <= x[f, t] * maxcapRMN for f in F for t in T)
    min_operating_RMN = model.addConstrs(quicksum(qEoLPa[v, f, t, p] for v in V for p in P) >= x[f, t] * molRMN for f in F for t in T)
    max_capa_H = model.addConstrs(quicksum(qRU[l, r, t, p] for r in R for p in P) <= x[l, t] * maxcapH for l in L for t in T)
    min_operating_H = model.addConstrs(quicksum(qRU[l, r, t, p] for r in R for p in P) >= x[l, t] * molH for l in L for t in T)
    max_capa_R = model.addConstrs(quicksum(qRP[r, m, t, p] for m in M for p in P) <= x[r, t] * maxcapR for r in R for t in T)
    min_operating_R = model.addConstrs(quicksum(qRP[r, m, t, p] for m in M for p in P) >= x[r, t] * molR for r in R for t in T)
    max_capa_DP = model.addConstrs(quicksum(qEoLP[m, r, t, p] for m in M for p in P) <= x[r, t] * maxcapDP for r in R for t in T)
    min_operating_DP = model.addConstrs(quicksum(qEoLP[m, r, t, p] for m in M for p in P) >= x[r, t] * molDP for r in R for t in T)
    max_capa_RF = model.addConstrs(quicksum(qRU[v, l, t, p] for l in L for p in P) <= x[v, t] * maxcapRF for v in V for t in T)
    min_operating_RF = model.addConstrs(quicksum(qRU[v, l, t, p] for l in L for p in P) >= x[v, t] * molRF for v in V for t in T)
    max_capa_DRU = model.addConstrs(quicksum(qEoLPa[v, f, t, p] for f in F for p in P) <= x[v, t] * maxcapDRU for v in V for t in T)
    min_operating_DRU = model.addConstrs(quicksum(qEoLPa[v, f, t, p] for f in F for p in P) >= x[v, t] * molDRU for v in V for t in T)
    
    flow_conservation_F = model.addConstrs((quicksum(qRU[f, l, t, p] for l in L) >= quicksum(qEoLPa[v, f, t, p] for v in V) for f in F for p in P for t in T), name="flow_conservation")
    flow_conservation_L = model.addConstrs((quicksum(qRU[f, l, t, p] for f in F) + quicksum(qRU[v, l, t, p] for v in V) + quicksum(qRU[l2, l, t, p] for l2 in [l1 for l1 in L if l1 != l]) == quicksum(qRU[l, r, t, p] for r in R) + quicksum(qRU[l, l2, t, p] for l2 in [l1 for l1 in L if l1 != l]) for l in L for p in P for t in T), name="flow_conservation_L")
    flow_conservation_R = model.addConstrs((quicksum(qRU[l, r, t, p] for l in L) == quicksum(qPR[m, r, t, p] for m in M) for r in R for p in P for t in T), name="flow_conservation_R")
    flow_conservation_RP = model.addConstrs((qRP[r, m, t, p] == qPR[m, r, t, p] for r in R for m in M for p in P for t in T), name="flow_conservation_RP")
    flow_conservation_RU = model.addConstrs((quicksum(qEoLP[m, r, t, p] for m in M) == quicksum(qEoLRU[r, v, t, p] for v in V) for r in R for p in P for t in T), name="flow_conservation_RU")
    flow_conservation_RF = model.addConstrs((quicksum(qRU[v, l, t, p] for l in L) + quicksum(qEoLPa[v, f, t, p] for f in F) == quicksum(qEoLRU[r, v, t, p] for r in R) for v in V for p in P for t in T), name="flow_conservation_RF")
    
    flow_authorisation_Trans = model.addConstrs((qTR[i, i2, t] <= Z * z[i, i2, t] for i, i2 in A_TR for t in T), name="flow_authorisation_Trans")
    flow_authorisation_M_RP = model.addConstrs((qRP[r, m, t, p] <= Z * z[m, r, t] for m in M for r in R for t in T for p in P), name="flow_authorisation_M")
    flow_authorisation_M_EoLP = model.addConstrs((qEoLP[m, r, t, p] <= Z * z[m, r, t] for m in M for r in R for t in T for p in P), name="flow_authorisation_M_EoLP")
    
    flow_authorisation_origin = model.addConstrs((z[i, i2, t] <= x[i, t] for t in T for i, i2 in A_TR), name="flow_authorisation_origin")
    flow_authorisation_destination = model.addConstrs((z[i, i2, t] <= x[i2, t] for t in T for i, i2 in A_TR), name="flow_authorisation_destination")
    flow_authorisation_retrofit = model.addConstrs((z[m, r, t] <= x[r, t] for m in M for r in R for t in T), name="flow_authorisation_retrofit")
    
    if collab == "Integrated" or collab == "Together":
        flow_authorisation_L = model.addConstrs((z[l, l2, t] == 0 for l in L for l2 in L if l != l2 for t in T), name="flow_authorisation_L")
    
    if collab == "Integrated":
        dedicated_logistics_paths_F = model.addConstrs((quicksum(z[f, l, t] for l in L) <= 1 for f in F for t in T), name="dedicated_logistics_paths_F")
        dedicated_logistics_paths_F_temp = model.addConstrs((z[f, l, t] == z[f, l, t + 1] for f in F for l in L for t in T[:-1]), name="dedicated_logistics_paths_F_temp")
        dedicated_logistics_paths_L = model.addConstrs((quicksum(z[l, r, t] for l in L) <= 1 for r in R for t in T), name="dedicated_logistics_paths_L")
        dedicated_logistics_paths_L_temp = model.addConstrs((z[l, r, t] == z[l, r, t + 1] for l in L for r in R for t in T[:-1]), name="dedicated_logistics_paths_L_temp")
        dedicated_logistics_paths_R = model.addConstrs((quicksum(z[r, v, t] for v in V) <= 1 for r in R for t in T), name="dedicated_logistics_paths_R")
        dedicated_logistics_paths_R_temp = model.addConstrs((z[r, v, t] == z[r, v, t + 1] for r in R for v in V for t in T[:-1]), name="dedicated_logistics_paths_R_temp")
        dedicated_logistics_paths_V = model.addConstrs((quicksum(z[v, l, t] for l in L) <= 1 for v in V for t in T), name="dedicated_logistics_paths_V")
        dedicated_logistics_paths_V_temp = model.addConstrs((z[v, l, t] == z[v, l, t + 1] for v in V for l in L for t in T[:-1]), name="dedicated_logistics_paths_V_temp")
    
        facility_activation = model.addConstrs((x[j, t] <= x[j, t + 1] for j in J for t in T[:-1]), name="facility_activation")
    
    if collab == "Together":
        facility_activation_F = model.addConstrs((x[j, t] <= x[j, t + 1] for j in F for t in T[:-1]), name="facility_activation_F")
        facility_activation_V = model.addConstrs((x[j, t] <= x[j, t + 1] for j in V for t in T[:-1]), name="facility_activation_V")
    
    for r in R:
        for m in M:
            if dist.loc[m[:-1], r[:-1]] <= D_max:
                for t in T:
                    model.addConstr(z[m, r, t] <= 1, name="distance_constraint")
            else:
                for t in T:
                    model.addConstr(z[m, r, t] == 0, name="distance_constraint")
    
    def set_param_optim(m, folder_path, timelimit=None):
        m.Params.MIPGap = 0.05
        if timelimit:
            m.Params.TimeLimit = timelimit
        m.setParam('OutputFlag', True)
        os.makedirs(folder_path, exist_ok=True)  # Ensure the folder exists
        log_file_path = os.path.join(folder_path, 'model.log')
        m.setParam('LogFile', log_file_path)
        return None
    
    set_param_optim(model, "experimentations/" + collab + "/" + current_time + "/")
    
    model.optimize()
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT or model.status == GRB.INTERRUPTED:
        optim_obj = {"Model Status": model.status, "Total footprint": model.objVal, "Activation": CF_FA.getValue(), "Operation": CF_OP.getValue(), "Transport": CF_TR.getValue(), "Lost orders": CF_LO.getValue()}
        df_result = pd.DataFrame(optim_obj.items(), columns=["Type", "Value"])
        for v in model.getVars():
            if v.x > 0:
                list_v = [v.varName, v.x]
                new_row = pd.Series(list_v, index=df_result.columns)
                df_result = pd.concat([df_result, new_row.to_frame().T], ignore_index=True)
        df_result.to_csv(destination_path+"results.csv", index=True)
