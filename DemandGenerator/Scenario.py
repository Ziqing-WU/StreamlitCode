from config import *

with st.sidebar:
    """
    Navigation

    [Scenario Analysis](#scenario-analysis)
    - [Select the Scenarios](#select-the-scenarios)
    - [Apply Bass Model](#apply-bass-model)

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

for scenario in scenarios:
    with open(f"scenarios/{scenario}/{current_year}_vehicle_list_TM.pickle", "rb") as f:
        data = pickle.load(f)
        df_TM = data["Dataframe"]
        parameters[scenario + "TM"] = data["Parameters"]
        df.loc["TM", scenario] = df_TM.shape[0]

    with open(f"scenarios/{scenario}/{current_year}_vehicle_list_TAM.pickle", "rb") as f:
        data = pickle.load(f)
        df_TAM = data["Dataframe"]
        parameters[scenario+ "TAM"] = data["Parameters"]
        df.loc["TAM", scenario] = df_TAM.shape[0]

    with open(f"scenarios/{scenario}/{current_year}_vehicle_list_SAM.pickle", "rb") as f:
        data = pickle.load(f)
        df_SAM = data["Dataframe"]
        parameters[scenario + "SAM"] = data["Parameters"]
        df.loc["SAM", scenario] = df_SAM.shape[0]

    with open(f"scenarios/{scenario}/{current_year}_vehicle_list_DEM.pickle", "rb") as f:
        data = pickle.load(f)
        df_DEM = data["Dataframe"]
        df.loc["DEM", scenario] = df_DEM[df_DEM["retrofit_service"]].shape[0]

# st.write(df)
# st.write(parameters)
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

# Show the plot
st.plotly_chart(fig)

st.write("## Apply Bass Model")
st.write(df_DEM.head(10))

df_vehicle = df_DEM[df_DEM["retrofit_service"]]
df_vehicle['retrofit_service'] = df_vehicle['retrofit_service'].astype(int)
df_demand = df_vehicle.groupby(by = ["code_commune_titulaire", "type_version_variante"]).agg({"retrofit_service": "sum"}).reset_index()

def bass_diff(N, p, q, M):
    return (p + q*N/M) * (M - N)

T_max = st.number_input("Enter the number of years to forecast", value=20)
T = np.arange(1, T_max + 1)
p = st.number_input("Enter the coefficient of innovation", value=0.03)
q = st.number_input("Enter the coefficient of imitation", value=0.38)
N0 = [0]
# N = odeint

