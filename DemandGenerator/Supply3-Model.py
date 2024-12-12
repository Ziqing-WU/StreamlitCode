from config import *

with st.sidebar:
    """
    Navigation

    [Model Formulation](#model-formulation)
    - [Objective Function](#objective-function)
    - [Constraints](#constraints)
    
    [Model Implementation](#model-implementation)
    """
st.title('Network Design Models')
st.write("# Model Formulation")
with open("parameters.pkl", "rb") as f:
    parameters = pickle.load(f)

collab, M, R, V, L, F, P, T, demand, pb_EoL, dist, maxcapMN, maxcapRMN, maxcapH, maxcapR, maxcapDP, maxcapRF, maxcapDRU, molMN, molRMN, molH, molR, molDP, molRF, molDRU, uofMN, uofRMN, uofH, uofR, uofDP, uofRF, uofDRU, wRU, wEoLPa, pl_TR, tf_TR, fr_TR, utf_PR, utf_RP, lofPR, lofEoLP, af, Z, D_max = parameters.values()

st.write(f"The collaborative strategy of the selected dataset is {collab}.")

# st.write(
#     """
#     ### Decision Variables


#     ## Objective Function

# """)

# st.write(
#         """
#     ## Constraints
    

#     """)

# if collab == "Integrated" or collab == "Together":
#     st.write("""

#     > **Warning:** In the *integrated* and *together* scenario, paths between logistics nodes are not permitted, leading to this constraint.

#     $$
#     \\begin{align}
#     z_{ll't} = 0 \\quad \\forall l \\in L, \\forall l' \\in L, \\forall t \\in T
#     \\end{align}
#     $$
#     """)

# if collab == "Integrated":  
#     st.write("""
#     ### Dedicated Logistics Paths

#     In the *integrated* scenario, a tightly controlled logistics strategy is adopted, where specific facilities are designated to interact exclusively with certain other facilities.

#     Each factory sends parts to at most one designated logistics node, with the assignment consistent across planning periods.

#     $$
#     \\begin{align}
#     \\sum_{l \\in L} z_{flt} \\leq 1 \\quad &\\forall f \\in F, \\forall t \\in T \\\\
#     z_{flt} = z_{fl(t+1)} \\quad &\\forall f \\in F, \\forall l \\in L, \\forall t \\in T \\backslash \\{|T|\\}
#     \\end{align}
#     $$

#     Each retrofit centre is served solely by a specific logistics node, with the assignment consistent across planning periods.

#     $$
#     \\begin{align}
#     \\sum_{l \\in L} z_{lrtt} \\leq 1 \\quad &\\forall r \\in R, \\forall t \\in T \\\\
#     z_{lrtt} = z_{lr(t+1)} \\quad &\\forall l \\in L, \\forall r \\in R, \\forall t \\in T \\backslash \\{|T|\\}
#     \\end{align}
#     $$

#     Each retrofit centre interacts with at most one designated recovery centre, with the assignment consistent across planning periods.

#     $$
#     \\begin{align}
#     \\sum_{v \\in V} z_{rvt} &\\leq 1 \\quad \\forall r \\in R, \\forall t \\in T \\\\
#     z_{rvt} &= z_{rv(t+1)} \\quad \\forall r \\in R, v \\in V, \\forall t \\in T \\backslash \\{|T|\\}
#     \\end{align}
#     $$

#     Each recovery centre sends items to at most one designated logistics node, with the assignment consistent across planning periods.

#     $$
#     \\begin{align}
#     \\sum_{l \\in L} z_{vlt} &\\leq 1 \\quad \\forall v \\in V, \\forall t \\in T \\\\
#     z_{vlt} &= z_{vl(t+1)} \\quad \\forall v \\in V, \\forall l \\in L, \\forall t \\in T \\backslash \\{|T|\\}
#     \\end{align}
#     $$
#     """)

# if collab == "Integrated":
#     st.write("""
#     **Warning:** Facility Activation

#     In the *integrated* scenario, once a facility is opened, it remains operational for the rest of the planning horizon.

#     $$
#     \\begin{align}
#     x_{jt} \\leq x_{j(t+1)} \\quad \\forall j \\in J, \\forall t \\in T \\backslash \\{|T|\\}
#     \\end{align}
#     $$
#     """)

# st.write("""
# ### Distance

# To ensure that market segments are only served by retrofit centres within an acceptable distance, the following constraints are imposed.

# $$
# \\begin{align}
# z_{mrt} \\leq \\mathbf{1}_{\\{d_{rm} \\leq D_{max}\\}} \\quad \\forall m \\in M, \\forall r \\in R, \\forall t \\in T
# \\end{align}
# $$

# ### Non-negativity

# All flow variables, lost order variables, and EoL demand variables are non-negative.

# $$
# \\begin{align}
# q_{ii'pt}^{RU}, q_{ii'pt}^{PR}, q_{ii'pt}^{RP}, q_{ii'pt}^{EoLP}, q_{ii'pt}^{EoLRU}, q_{ii'pt}^{EoLPa} &\\geq 0 \\\\
# lo^{PR}_{mpt}, lo^{EoLP}_{mpt} &\\geq 0 \\\\
# dm_{mpt}^{EoLP} &\\geq 0
# \\end{align}
# $$

# """)

st.write("# Model Implementation")

file_name = 'parameters.pkl'
current_time = datetime.now().strftime("%Y%m%d%H%M")
destination_path = "experimentations/" + collab + "/" + current_time + "/"
os.makedirs(destination_path, exist_ok=True) 
os.rename(file_name, destination_path+file_name)

demand = demand.set_index("code_commune_titulaire", drop=True)

def def_model(T, q_PR_tau=None, collab=collab, M=M, R=R, V=V, L=L, F=F, P=P, demand=demand, pb_EoL=pb_EoL, dist=dist, maxcapMN=maxcapMN, maxcapRMN=maxcapRMN, maxcapH=maxcapH, maxcapR=maxcapR, maxcapDP=maxcapDP, maxcapRF=maxcapRF, maxcapDRU=maxcapDRU, molMN=molMN, molRMN=molRMN, molH=molH, molR=molR, molDP=molDP, molRF=molRF, molDRU=molDRU, uofMN=uofMN, uofRMN=uofRMN, uofH=uofH, uofR=uofR, uofDP=uofDP, uofRF=uofRF, uofDRU=uofDRU, wRU=wRU, wEoLPa=wEoLPa, pl_TR=pl_TR, tf_TR=tf_TR, fr_TR=fr_TR, utf_PR=utf_PR, utf_RP=utf_RP, lofPR=lofPR, lofEoLP=lofEoLP, af=af, D_max=D_max):
    model = Model(collab)
    F = pd.Series(F).apply(lambda x: f"{x}F")
    L = pd.Series(L).apply(lambda x: f"{x}L")
    R = pd.Series(R).apply(lambda x: f"{x}R")
    V = pd.Series(V).apply(lambda x: f"{x}V")
    M = pd.Series(M).apply(lambda x: f"{x}M")

    I = pd.concat([F, L, R, V, M])
    J = pd.concat([F, L, R, V])
    A_FL = list(product(F, L))  # (F x L)
    A_LR = list(product(L, R))  # (L x R)
    A_RV = list(product(R, V))  # (R x V)
    A_VF = list(product(V, F))  # (V x F)
    A_VL = list(product(V, L))  # (V x L)
    # A_MR = list(product(M, R))  # (M x R)
    A_MR = []

    for r in R:
        for m in M:
            if dist.loc[m[:-1], r[:-1]] <= D_max:
                A_MR.append((m, r))

    A_RM = [(r, m) for m, r in A_MR]

    # Create the arcs (l, l') for l, l' in L where l != l'
    A_LL = [(l1, l2) for l1 in L for l2 in L if l1 != l2]

    # Combine all arcs to form A^TR
    A_TR = A_FL + A_LR + A_RV + A_VF + A_VL
    if collab == "Hyperconnected":
        A_TR = A_TR + A_LL
    A_RU = A_FL + A_LR + A_VL 
    if collab == "Hyperconnected":
        A_RU = A_RU + A_LL

    # Decision variables
    x = model.addVars(J, T, vtype=GRB.BINARY, name="x")

    if collab == "Integrated":
        z = model.addVars(A_TR, T, vtype=GRB.BINARY, name="z")

    qRU = model.addVars(A_RU, T, P, vtype=GRB.CONTINUOUS, name="qRU")
    qPR = model.addVars(A_MR, T, P, vtype=GRB.CONTINUOUS, name="qPR")
    qRP = model.addVars(A_RM, T, P, vtype=GRB.CONTINUOUS, name="qRP")

    qEoLP = model.addVars(A_MR, T, P, vtype=GRB.CONTINUOUS, name="qEoLP")
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
    if collab == "Hyperconnected":
        CF_OP_H = quicksum((quicksum(qRU[l, r, t, p] for r in R) + quicksum(qRU[l, l2, t, p] for l2 in [l1 for l1 in L if l1 != l])) * uofH for l in L for p in P for t in T)
    else:
        CF_OP_H = quicksum((quicksum(qRU[l, r, t, p] for r in R)) * uofH for l in L for p in P for t in T)
    CF_OP_R = quicksum(qRP[r, m, t, p] * uofR for (r, m) in A_RM for p in P for t in T) + quicksum(qEoLRU[r, v, t, p] * uofDP for v in V for r in R for p in P for t in T)
    CF_OP_RF = quicksum((quicksum(qRU[v, l, t, p] * uofRF for l in L) + quicksum(qEoLPa[v, f, t, p] * uofDRU for f in F)) for v in V for p in P for t in T)
    CF_OP = CF_OP_RMN + CF_OP_MN + CF_OP_H + CF_OP_R + CF_OP_RF

    CF_TR_Trans = quicksum(qTR[i, i2, t] * tf_TR * dist.loc[i[:-1], i2[:-1]] for i, i2 in A_TR for t in T)
    if utf_RP < 10e-4:
        CF_TR_V = quicksum(qPR[m, r, t, p] * utf_PR * dist.loc[m[:-1], r[:-1]] for (m, r) in A_MR for p in P for t in T) 
    else:
        CF_TR_V = quicksum(qPR[m, r, t, p] * utf_PR * dist.loc[m[:-1], r[:-1]] for (m, r) in A_MR for p in P for t in T) + quicksum(qRP[r, m, t, p] * utf_RP * dist.loc[r[:-1], m[:-1]] for (r, m) in A_RM for p in P for t in T) + quicksum(qEoLP[m, r, t, p] * utf_RP * dist.loc[m[:-1], r[:-1]] for (m, r) in A_MR for p in P for t in T)
    CF_TR = CF_TR_Trans + CF_TR_V

    CF_LO_R = quicksum(loPR[m, t, p] * lofPR for m in M for p in P for t in T)
    CF_LO_EoLP = quicksum(loEoLP[m, t, p] * lofEoLP for m in M for p in P for t in T)
    CF_LO = CF_LO_R + CF_LO_EoLP

    model.setObjective(CF_FA + CF_OP + CF_TR + CF_LO, GRB.MINIMIZE)
    # Map each m to a list of relevant r values
    R_for_m = defaultdict(list)
    for (m, r) in A_MR:
        R_for_m[m].append(r)
    M_for_r = defaultdict(list)
    for (m, r) in A_MR:
        M_for_r[r].append(m)
    # Constraints
    if collab ==  "Hyperconnected" and T[0] == 1:
        dm_recovery_1 = model.addConstrs((dmEoLP[m, 1, p] == 0 for m in M for p in P), name="dm_recovery_1")
    elif collab == "Integrated" or collab == "Together":
        dm_recovery_1 = model.addConstrs((dmEoLP[m, 1, p] == 0 for m in M for p in P), name="dm_recovery_1")
    
    if collab == "Integrated" or collab == "Together":
        dm_recovery_t = model.addConstrs(
        (
            dmEoLP[m, t, p] == quicksum(
                (
                    quicksum(qPR[m, r, tau, p] for r in R_for_m[m]) *
                    (pb_EoL[t - tau + 1] - pb_EoL[t - tau])
                )
                for tau in range(1, t)
            )
            for m in M for p in P for t in range(2, len(T) + 1)
        ),
        name="dm_recovery_t"
    )
    else:
        dm_recovery_t = model.addConstrs(
        (
            dmEoLP[m, t, p] == quicksum(
                (
                    quicksum(q_PR_tau[m, r, tau, p] for r in R_for_m[m]) *
                    (pb_EoL[t - tau + 1] - pb_EoL[t - tau])
                )
                for tau in range(1, t)
            )
            for m in M for p in P for t in T
        ),
        name="dm_recovery_t"
    )


    demand_fulfillment_PR = model.addConstrs((quicksum(qPR[m, r, t, p] for r in R_for_m[m]) == demand.loc[m[:-1], t] - loPR[m, t, p] for m in M for p in P for t in T), name="demand_fulfillment_PR")
    demand_fulfillment_EoLP = model.addConstrs((quicksum(qEoLP[m, r, t, p] for r in R_for_m[m]) == dmEoLP[m, t, p] - loEoLP[m, t, p] for m in M for p in P for t in T), name="demand_fulfillment_EoLP")

    transport_unit_flows_RU = model.addConstrs((quicksum(qRU[i, i2, t, p] * wRU for p in P) <= qTR[i, i2, t] * pl_TR * fr_TR for i, i2 in A_RU for t in T), name="transport_unit_flows")
    transport_unit_flows_EoLRU = model.addConstrs((quicksum(qEoLRU[i, i2, t, p] * wRU for p in P) <= qTR[i, i2, t] * pl_TR * fr_TR for i, i2 in A_RV for t in T), name="transport_unit_flows_EoLRU")
    transport_unit_flows_EoLPa = model.addConstrs((quicksum(qEoLPa[i, i2, t, p] * wEoLPa for p in P) <= qTR[i, i2, t] * pl_TR * fr_TR for i, i2 in A_VF for t in T), name="transport_unit_flows_EoLPa")

    max_capa_MN = model.addConstrs(quicksum(quicksum(qRU[f, l, t, p] for l in L) - quicksum(qEoLPa[v, f, t, p] for v in V) for p in P) <= x[f, t] * maxcapMN for f in F for t in T)
    if molMN > 10e-4:
        min_operating_MN = model.addConstrs(quicksum(quicksum(qRU[f, l, t, p] for l in L) - quicksum(qEoLPa[v, f, t, p] for v in V) for p in P) >= x[f, t] * molMN for f in F for t in T)
    max_capa_RMN = model.addConstrs(quicksum(qEoLPa[v, f, t, p] for v in V for p in P) <= x[f, t] * maxcapRMN for f in F for t in T)
    if molRMN > 10e-4:
        min_operating_RMN = model.addConstrs(quicksum(qEoLPa[v, f, t, p] for v in V for p in P) >= x[f, t] * molRMN for f in F for t in T)
    max_capa_H = model.addConstrs(quicksum(qRU[l, r, t, p] for r in R for p in P) <= x[l, t] * maxcapH for l in L for t in T)
    if molH > 10e-4:
        min_operating_H = model.addConstrs(quicksum(qRU[l, r, t, p] for r in R for p in P) >= x[l, t] * molH for l in L for t in T)
    max_capa_R = model.addConstrs(quicksum(qRP[r, m, t, p] for m in M_for_r[r] for p in P) <= x[r, t] * maxcapR for r in R for t in T)
    if molR > 10e-4:
        min_operating_R = model.addConstrs(quicksum(qRP[r, m, t, p] for m in M_for_r[r] for p in P) >= x[r, t] * molR for r in R for t in T)
    max_capa_DP = model.addConstrs(quicksum(qEoLP[m, r, t, p] for m in M_for_r[r] for p in P) <= x[r, t] * maxcapDP for r in R for t in T)
    if molDP > 10e-4:
        min_operating_DP = model.addConstrs(quicksum(qEoLP[m, r, t, p] for m in M_for_r[r] for p in P) >= x[r, t] * molDP for r in R for t in T)
    max_capa_RF = model.addConstrs(quicksum(qRU[v, l, t, p] for l in L for p in P) <= x[v, t] * maxcapRF for v in V for t in T)
    if molRF > 10e-4:
        min_operating_RF = model.addConstrs(quicksum(qRU[v, l, t, p] for l in L for p in P) + quicksum(qEoLPa[v, f, t, p] for f in F for p in P) >= x[v, t] * molRF for v in V for t in T)
    max_capa_DRU = model.addConstrs(quicksum(qEoLPa[v, f, t, p] for f in F for p in P) <= x[v, t] * maxcapDRU for v in V for t in T)
    # if molDRU > 10e-4:
    #     min_operating_DRU = model.addConstrs(quicksum(qEoLPa[v, f, t, p] for f in F for p in P) >= x[v, t] * molDRU for v in V for t in T)

    flow_conservation_F = model.addConstrs((quicksum(qRU[f, l, t, p] for l in L) >= quicksum(qEoLPa[v, f, t, p] for v in V) for f in F for p in P for t in T), name="flow_conservation")
    if collab == "Hyperconnected":
        flow_conservation_L = model.addConstrs((quicksum(qRU[f, l, t, p] for f in F) + quicksum(qRU[v, l, t, p] for v in V) + quicksum(qRU[l2, l, t, p] for l2 in [l1 for l1 in L if l1 != l]) == quicksum(qRU[l, r, t, p] for r in R) + quicksum(qRU[l, l2, t, p] for l2 in [l1 for l1 in L if l1 != l]) for l in L for p in P for t in T), name="flow_conservation_L")
    else:
        flow_conservation_L = model.addConstrs((quicksum(qRU[f, l, t, p] for f in F) + quicksum(qRU[v, l, t, p] for v in V) == quicksum(qRU[l, r, t, p] for r in R) for l in L for p in P for t in T), name="flow_conservation_L")

    flow_conservation_R = model.addConstrs((quicksum(qRU[l, r, t, p] for l in L) == quicksum(qPR[m, r, t, p] for m in M_for_r[r]) for r in R for p in P for t in T), name="flow_conservation_R")
    flow_conservation_RP = model.addConstrs((qRP[r, m, t, p] == qPR[m, r, t, p] for (r,m) in A_RM for p in P for t in T), name="flow_conservation_RP")
    flow_conservation_RU = model.addConstrs((quicksum(qEoLP[m, r, t, p] for m in M_for_r[r]) == quicksum(qEoLRU[r, v, t, p] for v in V) for r in R for p in P for t in T), name="flow_conservation_RU")
    flow_conservation_RF = model.addConstrs((quicksum(qRU[v, l, t, p] for l in L) + quicksum(qEoLPa[v, f, t, p] for f in F) == quicksum(qEoLRU[r, v, t, p] for r in R) for v in V for p in P for t in T), name="flow_conservation_RF")

    if collab == "Integrated":
        for i, i2 in A_TR:
            for t in T:
                model.addGenConstrIndicator(
                    z[i, i2, t],
                    0,
                    qTR[i, i2, t] == 0,
                    name=f"flow_authorisation_Trans_{i}_{i2}_{t}"
                )
        flow_authorisation_origin = model.addConstrs((z[i, i2, t] <= x[i, t] for t in T for i, i2 in A_TR), name="flow_authorisation_origin")
        flow_authorisation_destination = model.addConstrs((z[i, i2, t] <= x[i2, t] for t in T for i, i2 in A_TR), name="flow_authorisation_destination")

        dedicated_logistics_paths_F = model.addConstrs((quicksum(z[f, l, t] for l in L) <= 1 for f in F for t in T), name="dedicated_logistics_paths_F")
        dedicated_logistics_paths_F_temp = model.addConstrs((z[f, l, t] <= z[f, l, t + 1] for f in F for l in L for t in T[:-1]), name="dedicated_logistics_paths_F_temp")
        dedicated_logistics_paths_L = model.addConstrs((quicksum(z[l, r, t] for l in L) <= 1 for r in R for t in T), name="dedicated_logistics_paths_L")
        dedicated_logistics_paths_L_temp = model.addConstrs((z[l, r, t] <= z[l, r, t + 1] for l in L for r in R for t in T[:-1]), name="dedicated_logistics_paths_L_temp")
        dedicated_logistics_paths_R = model.addConstrs((quicksum(z[r, v, t] for v in V) <= 1 for r in R for t in T), name="dedicated_logistics_paths_R")
        dedicated_logistics_paths_R_temp = model.addConstrs((z[r, v, t] <= z[r, v, t + 1] for r in R for v in V for t in T[:-1]), name="dedicated_logistics_paths_R_temp")
        dedicated_logistics_paths_V = model.addConstrs((quicksum(z[v, l, t] for l in L) <= 1 for v in V for t in T), name="dedicated_logistics_paths_V")
        dedicated_logistics_paths_V_temp = model.addConstrs((z[v, l, t] <= z[v, l, t + 1] for v in V for l in L for t in T[:-1]), name="dedicated_logistics_paths_V_temp")

        facility_activation = model.addConstrs((x[j, t] <= x[j, t + 1] for j in J for t in T[:-1]), name="facility_activation")

    if collab == "Together":
        facility_activation_F = model.addConstrs((x[j, t] <= x[j, t + 1] for j in F for t in T[:-1]), name="facility_activation_F")
        facility_activation_V = model.addConstrs((x[j, t] <= x[j, t + 1] for j in V for t in T[:-1]), name="facility_activation_V")

    max_trans = 10000
    # (demand.loc[:,T].max().max() * wRU) / (pl_TR * fr_TR)
    qTR_border = model.addConstrs((qTR[i, i2, t]<= max_trans for (i, i2) in A_TR for t in T), name="qTR_border")
    
    q_PR_t = qPR
    return model, CF_FA, CF_OP, CF_TR, CF_LO, q_PR_t

def set_param_optim(m, folder_path, timelimit=None):
    # m.Params.MIPGap = 0.01
    if timelimit:
        m.Params.TimeLimit = timelimit
    m.setParam('OutputFlag', True)
    m.setParam('MIPFocus', 1)  # Focus on finding feasible solutions
    m.setParam("Heuristics", 0.5)
    m.setParam('Cuts', 2)  # Generate more cuts
    m.setParam('Presolve', 2)  # Aggressive presolve
    m.setParam('NumericFocus', 3)
    os.makedirs(folder_path, exist_ok=True)  # Ensure the folder exists
    log_file_path = os.path.join(folder_path, 'model.log')
    m.setParam('LogFile', log_file_path)
    return None

if collab == "Integrated" or collab == "Together":
    model, CF_FA, CF_OP, CF_TR, CF_LO, _ = def_model(T)
    # model =  model.relax()
    set_param_optim(model, "experimentations/" + collab + "/" + current_time + "/")
    model.optimize()
    model.write(destination_path + "model.rlp")
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT or model.status == GRB.INTERRUPTED:
        st.write("Optimal solution found with total cost", model.objVal)
        optim_obj = {"Model Status": model.status, "Total footprint": model.objVal, "Activation": CF_FA.getValue(), "Operation": CF_OP.getValue(), "Transport": CF_TR.getValue(), "Lost orders": CF_LO.getValue()}
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
else:
    q_PR_tau = {}
    for t in T:
        model, CF_FA, CF_OP, CF_TR, CF_LO, q_PR_t = def_model([t], q_PR_tau=q_PR_tau)
        set_param_optim(model, "experimentations/" + collab + "/" + current_time + "/")
        model.optimize()
        q_PR_t_values = model.getAttr('X', q_PR_t)
        q_PR_tau_list = [
            {'M': key[0], 'R': key[1], 't': key[2], 'p' : key[3], 'Value': value}
            for key, value in q_PR_tau.items()
        ]
        
        # Create a DataFrame
        df_q_PR_tau = pd.DataFrame(q_PR_tau_list)
        
        # Display the DataFrame
        st.dataframe(df_q_PR_tau)
        q_PR_tau.update(q_PR_t_values)
        if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT or model.status == GRB.INTERRUPTED:
            st.write("Optimal solution found with total cost", model.objVal)
            optim_obj = {"Model Status": model.status, "Total footprint": model.objVal, "Activation": CF_FA.getValue(), "Operation": CF_OP.getValue(), "Transport": CF_TR.getValue(), "Lost orders": CF_LO.getValue()}
            df_result = pd.DataFrame(optim_obj.items(), columns=["Type", "Value"])
            for v in model.getVars():
                if v.x > 0:
                    list_v = [v.varName, v.x]
                    new_row = pd.Series(list_v, index=df_result.columns)
                    df_result = pd.concat([df_result, new_row.to_frame().T], ignore_index=True)
            st.write(df_result)
            df_result.to_csv(destination_path+"results"+str(t)+".csv", index=True)
        else:
            st.write("No solution found")
        