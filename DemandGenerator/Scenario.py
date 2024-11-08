from config import *

with st.sidebar:
    """
    Navigation

    [Scenario Analysis](#scenario-analysis)
    - [Select the Scenarios](#select-the-scenarios)
    - [Apply Bass Model](#apply-bass-model)
    - [Apply Bass Model with different coefficients](#apply-bass-model-with-different-coefficients)
    - [Generate the demand for each scenario](#generate-the-demand-for-each-scenario)

    """
st.markdown(
    '''
    # Scenario Analysis
    Please create a folder called "scenario" and subfolders for each scenario, then put the corresponding data in the subfolders.
    '''
)
st.write("## Select the Scenarios")
scenarios = st.multiselect("Select the scenarios to be analyzed", os.listdir("scenarios"), default=os.listdir("scenarios"))
df = pd.DataFrame(columns=scenarios)
parameters = {}
df_TM = {}
df_TAM = {}
df_SAM = {}
df_DEM = {}
df_demand = {}

for scenario in scenarios:
    with open(f"scenarios/{scenario}/{current_year}_vehicle_list_TM.pickle", "rb") as f:
        data = pickle.load(f)
        df_TM[scenario] = data["Dataframe"]
        parameters[scenario + "TM"] = data["Parameters"]
        df.loc["TM", scenario] = df_TM[scenario].shape[0]

    with open(f"scenarios/{scenario}/{current_year}_vehicle_list_TAM.pickle", "rb") as f:
        data = pickle.load(f)
        df_TAM[scenario] = data["Dataframe"]
        parameters[scenario+ "TAM"] = data["Parameters"]
        df.loc["TAM", scenario] = df_TAM[scenario].shape[0]

    with open(f"scenarios/{scenario}/{current_year}_vehicle_list_SAM.pickle", "rb") as f:
        data = pickle.load(f)
        df_SAM[scenario] = data["Dataframe"]
        parameters[scenario + "SAM"] = data["Parameters"]
        df.loc["SAM", scenario] = df_SAM[scenario].shape[0]

    with open(f"scenarios/{scenario}/{current_year}_vehicle_list_DEM.pickle", "rb") as f:
        data = pickle.load(f)
        df_DEM[scenario] = data["Dataframe"]
        df.loc["DEM", scenario] = df_DEM[scenario][df_DEM[scenario]["retrofit_service"]].shape[0]

fig = make_subplots(rows=1, cols=len(scenarios), subplot_titles=scenarios, specs=[[{'type': 'funnel'}] * len(scenarios)])

for i, scenario in enumerate(scenarios, start=1):
    fig.add_trace(
        go.Funnel(
            y=df.index,
            x=df[scenario],
            textinfo="value+percent initial"
        ),
        row=1,
        col=i
    )

# Update layout
fig.update_layout(title='Funnel Charts for Each Scenario', showlegend=False)
fig.update_layout(
    # title_font_size=24,  # Adjust the title font size
    font=dict(size=18)  # Adjust the default font size for all text (including ticks)
)
# Show the plot
st.plotly_chart(fig)

st.write("## Apply Bass Model")
def bass_diff(N,t, p, q, M):
    return (p + q*N/M) * (M - N)
plot_total = go.Figure()
plot_total.update_layout(
    xaxis=dict(
        title="Year",
        title_font=dict(size=26),
        tickfont=dict(size=20)
    ),
    yaxis=dict(
        title="Total Demand",
        title_font=dict(size=26),
        tickfont=dict(size=20)
    ),
    legend=dict(
        font=dict(
            size=20
        )
    )
)

plot_increment = go.Figure()
plot_increment.update_layout(
    xaxis=dict(
        title="Year",
        title_font=dict(size=26),
        tickfont=dict(size=20)
    ),
    yaxis=dict(
        title="Demand per Year",
        title_font=dict(size=26),
        tickfont=dict(size=20)
    ),
    legend=dict(
        font=dict(
            size=20
        )
    )
)

T_max = st.number_input("Enter the number of years to forecast", value=20, key="maxi_year")
T = np.arange(0, T_max + 1)
p = st.number_input("Enter the coefficient of innovation", value=0.03)
q = st.number_input("Enter the coefficient of imitation", value=0.38)
N0 = [0]
for scenario in scenarios:
    df = df_DEM[scenario]
    df_vehicle = df[df["retrofit_service"]]
    df_vehicle['retrofit_service'] = df_vehicle['retrofit_service'].astype(int)
    df_demand[scenario] = df_vehicle.groupby(by = ["code_commune_titulaire", "type_version_variante"]).agg({"retrofit_service": "sum"}).reset_index()

    M_total = df_demand[scenario]["retrofit_service"].sum()
    N = odeint(bass_diff, N0, T, args=(p,q,M_total)).flatten()

    dN = np.diff(N)
    plot_total.add_trace(go.Scatter(x=T, y=N, mode='lines', name=scenario))
    plot_increment.add_trace(go.Scatter(x=T[1:], y=dN, mode='lines', name=scenario))

    
st.plotly_chart(plot_total)
st.plotly_chart(plot_increment)

st.write("## Apply Bass Model with different coefficients")
scenario = st.selectbox("Select the scenario to be analyzed", scenarios)
df_p_q = pd.DataFrame(columns=["p", "q"])
df_p_q.loc["baseline", :] = [0.03, 0.38]
df_p_q.loc["word-of-mouth dominant", :] = [0.01, 0.50]
df_p_q.loc["advertising dominant", :] = [0.05, 0.30]
st.write("The following coefficients are used for the Bass Model:", df_p_q)
plot_total_p_q = go.Figure()
plot_total_p_q.update_layout(
    xaxis=dict(
        title="Year",
        title_font=dict(size=26),
        tickfont=dict(size=20)
    ),
    yaxis=dict(
        title="Total Demand",
        title_font=dict(size=26),
        tickfont=dict(size=20)
    ),
    legend=dict(
        font=dict(
            size=20
        )
    )
)



plot_increment_p_q = go.Figure()
plot_increment_p_q.update_layout(
    xaxis=dict(
        title="Year",
        title_font=dict(size=26),
        tickfont=dict(size=20)
    ),
    yaxis=dict(
        title="Demand per Year",
        title_font=dict(size=26),
        tickfont=dict(size=20)
    ),
    legend=dict(
        font=dict(
            size=20
        )
    )
)
for index, row in df_p_q.iterrows():
    p = row["p"]
    q = row["q"]
    M_total = df_demand[scenario]["retrofit_service"].sum()
    N = odeint(bass_diff, N0, T, args=(p,q,M_total)).flatten()
    dN = np.diff(N)
    plot_total_p_q.add_trace(go.Scatter(x=T, y=N, mode='lines', name=index))
    plot_increment_p_q.add_trace(go.Scatter(x=T[1:], y=dN, mode='lines', name=index))

st.plotly_chart(plot_total_p_q)
st.plotly_chart(plot_increment_p_q)

st.write("## Generate the demand for each scenario")

if st.button("Download the demand for each scenario"):
    df_demand_p_q = {}
    for scenario in scenarios:
        for index, row in df_p_q.iterrows():
            p = row["p"]
            q = row["q"]
            for index,row in df_demand[scenario].iterrows():
                M = row["retrofit_service"]
                N0 = [0]
                N = odeint(bass_diff, N0, T, args=(p,q,M)).flatten()
                dN = np.diff(N)
                rounded_dN = np.round(dN).astype(int)
                rounded_sum = rounded_dN.sum()
                if rounded_sum > M:
                    # If sum exceeds market potential, decrement the largest values
                    excess = rounded_sum - M
                    sorted_indices = np.argsort(dN)
                    for idx in sorted_indices:
                        if excess <= 0:
                            break
                        if rounded_dN[idx] > 0:
                            rounded_dN[idx] -= 1
                            excess -= 1
                elif rounded_sum < M:
                    # If sum is less than market potential, increment the smallest values
                    deficit = M - rounded_sum
                    sorted_indices = np.argsort(-dN)
                    for idx in sorted_indices:
                        if deficit <= 0:
                            break
                        if rounded_dN[idx] > 0:
                            rounded_dN[idx] += 1
                            deficit -= 1
                df_demand[scenario].loc[index, np.arange(1, T_max + 1)] = rounded_dN
            df_demand_p_q[scenario + "_p=" + str(p) + "_q=" + str(q) ] = df_demand[scenario]
    
    with open("demand.pickle", "wb") as f:
        pickle.dump(df_demand_p_q, f)

