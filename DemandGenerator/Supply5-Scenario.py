from config import *

st.title('Scenario Analysis')

st.write("# Footprint Comparison")

scenarios = os.listdir("experimentations")

footprint = ["Activation", "Lost orders", "Transport","Operation", "Total footprint"]
dfs = []
for scenario in scenarios:
    for file in os.listdir(f"experimentations/{scenario}"):
        df = pd.read_csv(f"experimentations/{scenario}/{file}/results.csv", index_col=0)
        df = df[df["Type"].isin(footprint)]
        df_single_row = df.set_index("Type")["Value"].T
        df_single_row = pd.DataFrame([df_single_row])
        df_single_row.index = [file]
        dfs.append(df_single_row)

df = pd.concat(dfs)

Collab_order = ["Int", "Tog", "Hyp"]
Dynamics_order = ["WoM", "Base", "Adv"]

df["TVV"] = df.index.to_series().apply(lambda x: re.search(r'\d+', str(x)).group() if re.search(r'\d+', str(x)) else None)
df["Collaborative"] = df.index.str[:3]
df["Collaborative"] = pd.Categorical(df["Collaborative"], categories=Collab_order, ordered=True)
df["Dynamics"] = df.index.str.split("_").str[-1]
df["Dynamics"] = pd.Categorical(df["Dynamics"], categories=Dynamics_order, ordered=True)
df.sort_values(by=["TVV", "Collaborative", "Dynamics"], inplace=True)
st.write(df)

fig = px.bar(df, x=df.index, y=["Activation", "Lost orders", "Transport","Operation"], text_auto='.2s', barmode="stack")
fig.update_traces(textfont_size=30, textangle=0, textposition="inside", cliponaxis=False)

fig.update_layout(
    xaxis=dict(
        title="Scenarios",
        title_font=dict(size=26),
        tickfont=dict(size=20)
    ),
    yaxis=dict(
        title="Carbon Footprint",
        title_font=dict(size=26),
        tickfont=dict(size=20)
    ),
    legend=dict(
        title="Composition",
        title_font=dict(size=20),
        font=dict(
            size=20
        )
    )
)

st.plotly_chart(fig)
# Calculate percentages for annotation
df_percent = df[["Activation", "Lost orders", "Transport", "Operation"]].div(df["Total footprint"], axis=0) * 100
fig = px.bar(df_percent, x=df_percent.index, y=["Activation", "Lost orders", "Transport", "Operation"], text_auto=True)
st.plotly_chart(fig)

# Update layout for stacked bar chart

