from config import *

toc=["Settings", "Result Analysis", "Market Fulfillment", "Facility Location", "Transport Flow", "Product Flow for Retrofit and EoL", "Footprint Decomposition"]
st.sidebar.title("Navigation")
with st.sidebar:
    st.markdown("\n".join(f"- [{t}](#{t.replace(' ', '-').lower()})" for t in toc))
icons = {
        'L': 'warehouse',
        'R': 'wrench',
        'V': 'recycle',
        'F': 'industry'
    }
type_colors = {
    'L': 'blue',  # Blue for 'L'
    'R': 'red',   # Red for 'R'
    'V': 'green',  # Green for 'V'
    'F': 'orange'  # Orange for 'F'
}
st.header("Settings")
scenarios = os.listdir("experimentations")
scenario = st.selectbox("Select a collaborative strategy", scenarios)
entries = os.listdir(f"experimentations/{scenario}")
entry = st.selectbox("Select an experiment", entries)
folder_path = "experimentations/" + scenario + "/" + entry
pattern = os.path.join(folder_path, 'parameters*.pkl')
matched_files = glob.glob(pattern)
file_to_open = matched_files[0]
with open(file_to_open, "rb") as f:
    params = pickle.load(f)
# st.write(params)

st.header("Result Analysis")
px.set_mapbox_access_token("pk.eyJ1Ijoiend1LTE5IiwiYSI6ImNsNnc3a3Z3czA1dHUzY28xZjd6dzlvZDgifQ.BiY-VgoH8CR03_PckJLbpA")
file_path = folder_path + "/results.csv"

zoom_plotly = 6
zoom_folium = 7
height_map = 400


df_occitanie = get_communes_by_population(0)
# st.write(df_occitanie)


# if scenario == "Integrated" or scenario == "Together":
df_results = pd.read_csv(file_path, index_col=0)
st.write(df_results)
# else:
#     csv_files = glob.glob(folder_path+"/results*.csv")
#     df_results = pd.concat([pd.read_csv(f, index_col=0) for f in csv_files])

#     df_results = df_results.groupby("Type").sum().reset_index()
#     # df_results.to_csv(folder_path+"/results.csv")
#     st.write(df_results.head(2000))

'''
## Market Fulfillment
'''

# demand = params["demand"]
# # st.write("demand", demand)    

q_PR = df_results[df_results['Type'].str.startswith('qPR')].copy()
q_PR["M"] = q_PR["Type"].str.extract(r'qPR\[(\d+)')
q_PR["R"] = q_PR["Type"].str.extract(r'qPR\[\d+[A-Z],(\d+)')
q_PR["Year"] = q_PR["Type"].str.extract(r'qPR\[\d+[A-Z],\d+[A-Z],(\d+)')
q_PR["Product"] = q_PR["Type"].str.extract(r'qPR\[\d+[A-Z],\d+[A-Z],\d+,([a-zA-Z]+)\]')
# st.write("qPR", q_PR)

q_PR.drop(columns=["Type"], inplace=True)
dl_PR = q_PR.groupby(["M", "Product", "Year"]).sum().reset_index()
dl_PR = dl_PR.astype({"Year": int})
# st.write("dl_PR", dl_PR)

lo_PR = df_results[df_results['Type'].str.startswith('loPR')].copy()
lo_PR["M"] = lo_PR["Type"].str.extract(r'loPR\[(\d+)')
lo_PR["Year"] = lo_PR["Type"].str.extract(r'loPR\[\d+[A-Z],(\d+)')
lo_PR["Product"] = lo_PR["Type"].str.extract(r'loPR\[\d+[A-Z],\d+,(.*?)\]')
lo_PR.drop(columns=["Type"], inplace=True)
# st.write("lo_PR", lo_PR)

dl_PR['Year'] = dl_PR['Year'].astype(int)
lo_PR['Year'] = lo_PR['Year'].astype(int)

dl_PR_agg = dl_PR.groupby(['M', 'Product', 'Year']).sum().reset_index()
lo_PR_agg = lo_PR.groupby(['M', 'Product', 'Year']).sum().reset_index()
# st.write("dl_PR_agg", dl_PR_agg)

# Merge dl_PR and lo_PR on ['M', 'Product', 'Year'] using an outer join
dm_PR = pd.merge(
    dl_PR[['M', 'Product', 'Year', 'Value']],
    lo_PR[['M', 'Product', 'Year', 'Value']],
    on=['M', 'Product', 'Year'],
    how='outer',
    suffixes=('_dl', '_lo')
)

# Fill NaN values with 0 and compute the total 'Value'
dm_PR['Value'] = dm_PR['Value_dl'].fillna(0) + dm_PR['Value_lo'].fillna(0)

# Select the necessary columns
dm_PR = dm_PR[['M', 'Product', 'Year', 'Value']]

# Sort the DataFrame
dm_PR.sort_values(['M', 'Product', 'Year'], inplace=True)
dmEoLP = df_results[df_results['Type'].str.startswith('dmEoLP')].copy()

dmEoLP["M"] = dmEoLP["Type"].str.extract(r'dmEoLP\[(\d+)')
dmEoLP["Product"] = dmEoLP["Type"].str.extract(r'dmEoLP\[\d+[A-Z],\d+,(.*?)\]')
dmEoLP["Year"] = dmEoLP["Type"].str.extract(r'dmEoLP\[\d+[A-Z],(\d+)')
dmEoLP.drop(columns=["Type"], inplace=True)
dmEoLP = dmEoLP.merge(df_occitanie, left_on="M", right_on="COM")
dmEoLP.sort_values(by=["PMUN"], inplace=True)



loEoLP = df_results[df_results['Type'].str.startswith('loEoLP')].copy()
loEoLP["M"] = loEoLP["Type"].str.extract(r'loEoLP\[(\d+)')
loEoLP["Product"] = loEoLP["Type"].str.extract(r'loEoLP\[\d+[A-Z],\d+,(.*?)\]')
loEoLP["Year"] = loEoLP["Type"].str.extract(r'loEoLP\[\d+[A-Z],(\d+)')
loEoLP.drop(columns=["Type"], inplace=True)
# st.write("dm_PR", dm_PR)

if scenario == "Integrated" or scenario == "Together":
    dm_PR = dm_PR.merge(df_occitanie, left_on="M", right_on="COM")
    dm_PR.sort_values(by=["PMUN"], inplace=True)
    fig_bar = px.bar(dm_PR, x="Year", y="Value", color="Commune", title="Demand for retrofit per city")
    fig_bar.update_layout(showlegend=False)
    st.plotly_chart(fig_bar)

    dl_PR["Type"] = "Delivered"
    lo_PR["Type"] = "Lost"
    lo_PR = lo_PR.astype({"Year": int})
    PR = pd.concat([dl_PR, lo_PR])
    PR = PR.merge(df_occitanie, left_on="M", right_on="COM")
    st.plotly_chart(px.bar(PR, x="Year", y="Value", color="Type", hover_data="Commune", title="Fulfillment for retrofit"))
    st.write(f"The lost retrofit order is {lo_PR['Value'].sum()}")
    year_PR = st.slider("Year", min_value=1, max_value=20, value=1, step=1, key="year_PR")
    fig = px.scatter_mapbox(PR[PR["Year"]==year_PR], lat="Latitude", lon="Longitude", color="Type", size="Value", hover_data=["Commune", "Value"],color_discrete_sequence=px.colors.qualitative.Dark2, zoom=zoom_plotly, height=height_map, title="Geographical distribution of retrofit fulfillment")
    st.plotly_chart(fig)
    
    fig_bar_eol = px.bar(dmEoLP, x="Year", y="Value", color="Commune", title="Demand for EoL products")
    fig_bar_eol.update_layout(showlegend=False)
    st.plotly_chart(fig_bar_eol)

    # Demand for EoL products
    
    # st.write(dmEoLP, loEoLP)

    dlEoLP = dmEoLP[['M', 'Product', 'Year']].copy()
    for row in dlEoLP.iterrows():
        lo = loEoLP[(loEoLP["M"]==row[1]["M"]) & (loEoLP["Product"]==row[1]["Product"]) & (loEoLP["Year"]==row[1]["Year"])]
        if lo.shape[0] == 0:
            dlEoLP.loc[row[0], "Value"] = dmEoLP[(dmEoLP["M"]==row[1]["M"]) & (dmEoLP["Product"]==row[1]["Product"]) & (dmEoLP["Year"]==row[1]["Year"])].Value.values[0]
        else:
            dlEoLP.loc[row[0], "Value"] = dmEoLP[(dmEoLP["M"]==row[1]["M"]) & (dmEoLP["Product"]==row[1]["Product"]) & (dmEoLP["Year"]==row[1]["Year"])].Value.values[0] - lo.Value.values[0]
    # st.write(dlEoLP)
    dlEoLP["Type"] = "Delivered"
    loEoLP["Type"] = "Lost"
    EoLP = pd.concat([dlEoLP, loEoLP])
    EoLP = EoLP.merge(df_occitanie, left_on="M", right_on="COM")
    st.write(f"The number of EoL take-back order is {dmEoLP['Value'].sum()}")
    bar = px.bar(EoLP, x="Year", y="Value", color="Type", hover_data='Commune', title="Fulfillment for EoL products")
    bar.update_layout(yaxis_title="Number of EoL Recovery Orders")
    st.plotly_chart(bar)
    st.write(f"The number of lost EoL take-back order is {loEoLP["Value"].sum()}")

    year_EoLP = st.slider("Year", min_value=1, max_value=20, value=1, step=1, key="year_EoLP")
    st.plotly_chart(px.scatter_mapbox(EoLP[EoLP["Year"]==str(year_EoLP)], lat="Latitude", lon="Longitude", color="Type", size="Value", hover_data=["Commune", "Value"],color_discrete_sequence=px.colors.qualitative.Dark2, zoom=zoom_plotly, height=height_map, title="Geographical distribution of EoL fulfillment"))

else:
    st.write(f"The number of lost retrofit orders is {lo_PR['Value'].sum()}")
    st.write(f"The number of EoL take-back orders is {dmEoLP['Value'].sum()}")
    st.write(f"The number of lost EoL take-back orders is {loEoLP['Value'].sum()}")
    # dm_PR = dm_PR.merge(df_occitanie, left_on="M", right_on="COM")
    # dm_PR.sort_values(by=["PMUN"], inplace=True)
    # fig_bar = px.bar(dm_PR, x="Year", y="Value", color="Commune", title="Demand for retrofit per city")
    # fig_bar.update_layout(showlegend=False)
    # # st.plotly_chart(fig_bar)

    # dl_PR["Type"] = "Delivered"
    # lo_PR["Type"] = "Lost"
    # lo_PR = lo_PR.astype({"Year": int})
    # PR = pd.concat([dl_PR, lo_PR])
    # PR = PR.merge(df_occitanie, left_on="M", right_on="COM")
    # # st.plotly_chart(px.bar(PR, x="Year", y="Value", color="Type", hover_data="Commune", title="Fulfillment for retrofit"))
    # # year_PR = st.slider("Year", min_value=1, max_value=20, value=1, step=1, key="year_PR")
    # # fig = px.scatter_mapbox(PR[PR["Year"]==year_PR], lat="Latitude", lon="Longitude", color="Type", size="Value", hover_data=["Commune", "Value"],color_discrete_sequence=px.colors.qualitative.Dark2, zoom=zoom_plotly, height=height_map, title="Geographical distribution of retrofit fulfillment")
    # # st.plotly_chart(fig)

    dmEoLP = df_results[df_results['Type'].str.startswith('dmEoLP')].copy()
    # st.write(dmEoLP)
    loEoLP = df_results[df_results['Type'].str.startswith('loEoLP')].copy()
    dmEoLP["M"] = dmEoLP["Type"].str.extract(r'dmEoLP\[(\d+)')
    dmEoLP["Product"] = dmEoLP["Type"].str.extract(r'dmEoLP\[\d+[A-Z],\d+,(.*?)\]')
    dmEoLP["Month"] = dmEoLP["Type"].str.extract(r'dmEoLP\[\d+[A-Z],(\d+)')
    dmEoLP.drop(columns=["Type"], inplace=True)
    dmEoLP_month = dmEoLP.groupby(["Month", "Product"]).sum().reset_index()
    dmEoLP_month.drop(columns=["M"], inplace=True)
    # dmEoLP = dmEoLP.merge(df_occitanie, left_on="M", right_on="COM")
    # dmEoLP.sort_values(by=["PMUN"], inplace=True)
    # st.write(dmEoLP_month)
    dmEoLP_month_line = px.scatter(dmEoLP_month, x="Month", y="Value", title="Demand for EoL products", color_discrete_sequence=["darkblue"])
    dmEoLP_month_line.update_traces(name="EoL Demand", showlegend=True)
    # # fig_bar_eol = px.bar(dmEoLP, x="Year", y="Value", title="Demand for EoL products")
    # # fig_bar_eol.update_layout(showlegend=False)
    # # st.plotly_chart(fig_bar_eol)


    loEoLP["M"] = loEoLP["Type"].str.extract(r'loEoLP\[(\d+)')
    loEoLP["Product"] = loEoLP["Type"].str.extract(r'loEoLP\[\d+[A-Z],\d+,(.*?)\]')
    loEoLP["Month"] = loEoLP["Type"].str.extract(r'loEoLP\[\d+[A-Z],(\d+)')
    loEoLP.drop(columns=["Type"], inplace=True)
    loEoLP = loEoLP.groupby(["Month", "Product"]).sum().reset_index()
    loEoLP.drop(columns=["M"], inplace=True)
    # st.write(loEoLP)
    fig_scatter = px.scatter(loEoLP, x="Month", y="Value", title="Lost orders for EoL products", color_discrete_sequence=["red"])
    fig_scatter.update_traces(name="Lost orders", showlegend=True)
    dmEoLP_month_line.add_traces(fig_scatter.data)
    st.plotly_chart(dmEoLP_month_line)
    # dlEoLP = dmEoLP[['M', 'Product', 'Year']].copy()
    # for row in dlEoLP.iterrows():
    #     lo = loEoLP[(loEoLP["M"]==row[1]["M"]) & (loEoLP["Product"]==row[1]["Product"]) & (loEoLP["Year"]==row[1]["Year"])]
    #     if lo.shape[0] == 0:
    #         dlEoLP.loc[row[0], "Value"] = dmEoLP[(dmEoLP["M"]==row[1]["M"]) & (dmEoLP["Product"]==row[1]["Product"]) & (dmEoLP["Year"]==row[1]["Year"])].Value.values[0]
    #     else:
    #         dlEoLP.loc[row[0], "Value"] = dmEoLP[(dmEoLP["M"]==row[1]["M"]) & (dmEoLP["Product"]==row[1]["Product"]) & (dmEoLP["Year"]==row[1]["Year"])].Value.values[0] - lo.Value.values[0]
    # # st.write(dlEoLP)
    # dlEoLP["Type"] = "Delivered"
    # loEoLP["Type"] = "Lost"
    # # st.write(f"The lost EoL take-back order is {loEoLP["Value"].sum()}")
    # EoLP = pd.concat([dlEoLP, loEoLP])
    # EoLP = EoLP.merge(df_occitanie, left_on="M", right_on="COM")
    # EoLP = EoLP.groupby(["Product", "Year", "Type"]).sum().reset_index()
    # st.write(EoLP)
    # # st.plotly_chart(px.bar(EoLP, x="Year", y="Value", color="Type", hover_data='Commune', title="Fulfillment for EoL products"))


    
'''
## Facility Location
'''
# st.write(df_results[df_results['Type'].str.startswith('z')])

x = df_results[df_results['Type'].str.startswith('x')].copy()
x["COM"] = x["Type"].str.extract(r'x\[(\d+)')
x["year"] = x["Type"].str.extract(r'x\[\d+[FLRV],(\d+)\]')
x["type"] = x["Type"].str.extract(r'x\[\d+([FLRV]),\d+\]')
st.write(x)
x.drop(columns=["Type", "Value"], inplace=True)
x = x.merge(df_occitanie, on="COM")
np.random.seed(0)

for commune in x["Commune"].unique():
    for type in x["type"].unique():
        x.loc[(x["Commune"]==commune) & (x["type"]==type), "Longitude"] = x[(x["Commune"]==commune) & (x["type"]==type)]["Longitude"] + np.random.normal(0, 0.02)
        x.loc[(x["Commune"]==commune) & (x["type"]==type), "Latitude"] = x[(x["Commune"]==commune) & (x["type"]==type)]["Latitude"] + np.random.normal(0, 0.02)
if scenario == "Integrated" or scenario == "Together":
    year_filt = st.slider("Year", min_value=1, max_value=20, value=1, step=1)
    types = st.multiselect("Type of facilities", x["type"].unique(), default=x["type"].unique())
    x_year = x[x["year"]==str(year_filt)]
    x_year = x_year[x_year["type"].isin(types)]
    st.write(x_year)
else:
    year_filt = st.slider("Month", min_value=1, max_value=240, value=1, step=1)
    types = st.multiselect("Type of facilities", x["type"].unique(), default=x["type"].unique())
    x_year = x[x["year"]==str(year_filt)]
    x_year = x_year[x_year["type"].isin(types)]
    st.write(x_year)

st.write(x)
if scenario == "Integrated" or scenario == "Together":
    T = 20
else:
    T = 240

years = list(range(1, T+1))
years = [str(year) for year in years]
x_transpose = pd.DataFrame(columns=["type", "COM"] + years)

rows = []
for type in ["R", "V", "L", "F"]:
    COM = x[x["type"]==type]["COM"].unique()
    for com in COM:
        row = {"type": type, "COM": com}
        row.update({year: 0 for year in years})
        present_years = x[
            (x["type"] == type) & (x["COM"] == com)
        ]["year"].tolist()
        for year in present_years:
            row[year] = 1
        rows.append(row)
x_transpose = pd.DataFrame(rows)
x_transpose.fillna(0, inplace=True)
# st.write(x_transpose)
timeline = x_transpose.groupby("type").sum()
timeline.drop(columns=["COM"], inplace=True)

# st.write(timeline)

fig = go.Figure()
# Add a line for each type

for type_ in ["R", "V", "L", "F"]:
    fig.add_trace(go.Scatter(
        x=timeline.columns, 
        y=timeline.loc[type_], 
        mode='markers', 
        name=f'{type_}',
        marker=dict(color=type_colors[type_], opacity=0.6, size=5),
    ))
    # st.write(timeline.loc[type_])
fig.update_layout(
    xaxis_title="Planning Periods",
    yaxis_title="Number of facilities",
    legend_title="Type",
)
st.plotly_chart(fig)


if x_year.shape[0] == 0:
    st.write("No facilities found.")
    
else:
    center_lat = x_year.Latitude.mean()
    center_lon = x_year.Longitude.mean()

    # st.write(x_year)
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_folium, height=height_map)

    # Add markers
    for index, row in x_year.iterrows():
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=f"{row['Commune']}:\nPopulation {row['PMUN']}",
            icon=folium.Icon(icon=icons[row['type']], prefix='fa', color=type_colors[row['type']])
        ).add_to(m)

    folium_static(m)

if scenario == "Integrated" or scenario == "Together":
    st.write(f"PR at year {year_filt}",PR[PR["Year"]==year_filt])
    fig = px.scatter_mapbox(PR[PR["Year"]==year_filt],
                         lat="Latitude",
                         lon="Longitude",
                         color="Type",
                         size="Value",
                         hover_data=["Commune", "Value"],
                         color_discrete_sequence=px.colors.qualitative.Dark2,
                         mapbox_style="carto-positron",
                         zoom=zoom_plotly,
                         height=height_map,
                         title="Geographical distribution of retrofit fulfillment")
    size_icon = 14
    for icon in icons:
        fig.add_trace(go.Scattermapbox(
        lat=x_year[x_year["type"]==icon]["Latitude"],
        lon=x_year[x_year["type"]==icon]["Longitude"],
        mode='markers',
        marker=go.scattermapbox.Marker(size=size_icon, color=type_colors[icon]),
        name=icon,
        showlegend=True
    ))

    st.plotly_chart(fig)



'''
## Transport Flow
Here we show the transport flow situation among factories, logistics nodes, retrofit centers and recovery centers.
'''

qTR = df_results[df_results['Type'].str.startswith('qTR')].copy()
qTR["O"] = qTR["Type"].str.extract(r'qTR\[(\d+)')
qTR["type_O"] = qTR["Type"].str.extract(r'qTR\[\d+([A-Z])')
qTR["D"] = qTR["Type"].str.extract(r'qTR\[\d+[A-Z],(\d+)')
qTR["type_D"] = qTR["Type"].str.extract(r'qTR\[\d+[A-Z],\d+([A-Z])')
qTR["Year"] = qTR["Type"].str.extract(r'qTR\[\d+[A-Z],\d+[A-Z],(\d+)\]')

qTR = qTR.merge(df_occitanie.add_suffix("_O"), left_on="O", right_on="COM_O")
qTR = qTR.merge(df_occitanie.add_suffix("_D"), left_on="D", right_on="COM_D")
qTR = qTR[["O", "Commune_O", "Latitude_O", "Longitude_O", "type_O", "D", "Commune_D", "Latitude_D", "Longitude_D", "type_D", "Year", "Value"]]
max_qTR = max(qTR["Value"])
year_qTR = st.slider("Year", min_value=1, max_value=20, value=1, step=1, key="year_qTR")
qTR_year = qTR[qTR["Year"]==str(year_qTR)]
st.write(qTR_year[["Commune_O", "type_O", "Commune_D", "type_D", "Value"]])
show_icon = st.checkbox("Show icon", value=False)

m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_folium, height=height_map)

for index, row in qTR_year.iterrows():
    origin = [row['Latitude_O'], row['Longitude_O']]
    destination = [row['Latitude_D'], row['Longitude_D']]
    weight = row['Value']/max_qTR*50
    if show_icon:
        folium.Marker(
            location=[row['Latitude_O'], row['Longitude_O']],
            popup=f"{row['Commune_O']}:\n{row['type_O']}",
            icon=folium.Icon(icon=icons[row['type_O']], prefix='fa', color=type_colors[row['type_O']])
        ).add_to(m)
        folium.Marker(
            location=[row['Latitude_D'], row['Longitude_D']],
            popup=f"{row['Commune_D']}:\n{row['type_D']}",
            icon=folium.Icon(icon=icons[row['type_D']], prefix='fa', color=type_colors[row['type_D']])
        ).add_to(m)

    folium.PolyLine(
        locations=[origin, destination],
        color='black',
        weight=weight,
        tooltip=f"{row['Value']} transport units from {row['Commune_O']} {row['type_O']} to {row['Commune_D']} {row['type_D']}"
    ).add_to(m)

folium_static(m)

'''
## Product Flow for Retrofit and EoL
'''
qPR = pd.concat([df_results[df_results['Type'].str.startswith('qPR')]
                , df_results[df_results['Type'].str.startswith('qEoLP')]]).copy()
qPR["type"] = qPR["Type"].str.extract(r'q([A-Za-z]+)\[[^\]]*\]')
qPR["M"] = qPR["Type"].str.extract(r'q[A-Za-z]+\[(\d+)')
qPR["R"] = qPR["Type"].str.extract(r'q[A-Za-z]+\[\d+[A-Z],(\d+)')
qPR["Year"] = qPR["Type"].str.extract(r'q[A-Za-z]+\[\d+[A-Z],\d+[A-Z],(.*?),')
qPR["Product"] = qPR["Type"].str.extract(r'q[A-Za-z]+\[\d+[A-Z],\d+[A-Z],.*?,(.*?)\]')
qPR = qPR.merge(df_occitanie.add_suffix("_M"), left_on="M", right_on="COM_M")
qPR = qPR.merge(df_occitanie.add_suffix("_R"), left_on="R", right_on="COM_R")
st.write(qPR)
year_qPR = st.slider("Year", min_value=1, max_value=20, value=1, step=1, key="year_qPR")
m = folium.Map(location=[center_lat, center_lon], zoom_start=7, height=height_map)
qPR_year = qPR[qPR["Year"]==str(year_qPR)]
st.write(qPR_year[["Commune_M", "Commune_R", "Product", "Value", "type"]])
show_icon = st.checkbox("Show icon", value=False, key="icon_pd")
for index, row in qPR_year.iterrows():
    origin = [row['Latitude_M'], row['Longitude_M']]
    destination = [row['Latitude_R'], row['Longitude_R']]
    type = row['type']
    weight = row['Value']/max_qTR*50

    if show_icon:
        folium.Marker(
            location=[row['Latitude_M'], row['Longitude_M']],
            popup=f"{row['Commune_M']}",
            icon=folium.Icon(icon='user', prefix='fa')
        ).add_to(m)
        folium.Marker(
            location=[row['Latitude_R'], row['Longitude_R']],
            popup=f"{row['Commune_R']}",
            icon=folium.Icon(icon=icons['R'], prefix='fa', color=type_colors['R'])
        ).add_to(m)
    if type == "PR":
        color = 'black'
    else: 
        color = 'green'
    folium.PolyLine(
        locations=[origin, destination],
        color=color,
        weight=weight,
        tooltip=f"{row['Value']} {row['type']} from {row['Commune_M']} to {row['Commune_R']}"
    ).add_to(m)

folium_static(m)

'''
## Footprint Decomposition
'''

footprint = ["Activation", "Lost orders", "Transport","Operation"]
decomp = df_results[df_results['Type'].isin(footprint)].copy()
# st.write(decomp, df_results[df_results['Type']=="Total footprint"].Value.values[0])
decomp["Percentage"] = decomp["Value"]/df_results[df_results['Type']=="Total footprint"].Value.values[0]
st.plotly_chart(px.treemap(decomp, path=['Type'], values='Percentage', title="Footprint decomposition"))

