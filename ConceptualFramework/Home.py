import streamlit as st
import plotly.graph_objects as go
import random

st.markdown("Sankey Diagram to link PI and CSC with HCSC")

label = ["HCSC " + str(i+1) for i in range(9)] + ["CSC " + str(i+1) for i in range(6)] + ["PI " + str(i+1) for i in range(13)]
label[0] = "<b>Design products for circularity"
label[1] = "<b>Materialize objects on-demand <br>in PI open production fabs"
label[2] = "<b>Recover materials and energy <br>as locally as possible"
label[3] = "<b>Deliver materials and products <br>with hyperconnected logistics system"
label[4] = "<b>Enable sharing economy with PI"
label[5] = "<b>Exploit new circular functionalities<br> of existing facilities"
label[6] = "<b>Deploy open and hyperconnected <br>sustainability performance monitoring"
label[7] = "<b>Embrace technology innovation"
label[8] = "<b>Stimulate business model innovation"
label[9:15] = ["R-imperatives", "Restorative & regenerative cycles", "Sustainability framework", "Value focus", "Holistic system thinking", "Paradigm shift"]
label[15] = "Encapsulate merchandise in world standard green modular containers"
label[16] = "Aiming toward universal interconnectivity"
label[17] = "Evolve from material to PI-container handling and storage systems"
label[18] = "Exploit smart networked containers embedding smart objects"
label[19] = "Evolve from point-to-point hub-and-spoke transport <br>to distributed multi-segment intermodal transport"
label[20] = "Embrace a unified multi-tier conceptual framework"
label[21] = "Activate & exploit an open global supply web"
label[22] = "Design products fitting containers with minimal space waste"
label[23] = "Minimize physical moves and storages by digitally transmitting knowledge <br>and materializing objects as locally as possible"
label[24] = "Deploy open performance monitoring and capability certifications"
label[25] = "Prioritize webbed reliability and resilience of networks"
label[26] = "Stimulate business model innovation"
label[27] = "Enable open infrastructural innovation"

x = [0.45 for i in range(9)] + [0.999 for i in range(6)] + [0.001 for i in range(13)]
y = [i/9+0.001 for i in range(9)] + [i/6+0.001 for i in range(6)] + [i/13+0.001 for i in range(13)]

source = [0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6, 6, 7, 7, 8, 8, 22,21,23,16,23,25,16,17,19,20,21,15,16,18,26,27,24,27,26]
target = [9,10,12,9, 12,9, 10, 10,13, 14,12,11,9,11,12,11,13,9,10,13,14, 0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 5, 5, 6, 7, 8]
value =  [2, 2, 6, 6, 4, 5, 5, 4, 4, 2, 5, 5, 2, 4, 4, 7, 3, 5, 5, 5, 5, 10,5, 5, 2, 6, 2, 2, 2, 2, 2, 2, 4, 2, 4, 5, 5, 10,10,10]

color_link = list(range(len(source)))
# color_list = ["#4874c5", "#6d89d2", "#95a2df", "#b7c1e8", "#d6e0f0", "#a0c6e0", "#7baed1", "#5996c2", "#417eb3"]
# color_list = ["#4874C5", "#9DB7DA", "#77ABBD", "#f2e9e4","#8ecae6","#A8DADC", "#edddd4","#457B9D","#DCE9E9"]
# color_list = ['rgba(255, 255, 255, 0)', 'rgba(255, 255, 255, 0)', "#4874C5", "#4874C5", 'rgba(255, 255, 255, 0)', 'rgba(255, 255, 255, 0)', 'rgba(255, 255, 255, 0)', 'rgba(255, 255, 255, 0)', 'rgba(255, 255, 255, 0)']
color_list = ["#395CB4", "#7C94C3", "#5D8D9A", "#dacbbc", "#71b0d9", "#88B0B0", "#dab7ac", "#38627C", "#B0C3C3"]


for j in range(len(source)):
    for i in range(9):
        if source[j] == i or target[j] == i:
            color_link[j] = color_list[i]

st.write(color_link)
            
# color_link = ['#E5E8F7' if (target[i] in {0, 2, 4, 6, 8}) or (source[i] in {0, 2, 4, 6, 8}) else '#beceea' for i in range(len(source))]
color_node = '#0e182c'
# color_node = '#ffffff'

link = dict(source = source, target = target, value = value, color = color_link)
node = dict(x = x, y = y, label = label, pad=28, thickness=5, color = color_node)
node['line'] = dict(color='black', width=0)
data = go.Sankey(link = link, node=node)# plot
fig = go.Figure(data)
fig.update_layout(font=dict(size=18, family="Arial", color = '#0e182c'))

x_coordinates = [0.001, 0.53, 0.999]
for i, column_name in enumerate(["<i>PI characteristics","<i>HCSC characteristics","<i>CSC characteristics"]):
  fig.add_annotation(
          x=x_coordinates[i],
          y=1.1,
          text=column_name,
          showarrow=False,
          align="left",
          font=dict(size=22)
          )
  
fig.update_layout(
xaxis={
'showgrid': False, # thin lines in the background
'zeroline': False, # thick line at x=0
'visible': False,  # numbers below
},
yaxis={
'showgrid': False, # thin lines in the background
'zeroline': False, # thick line at x=0
'visible': False,  # numbers below
}, plot_bgcolor='rgba(0,0,0,0)')





st.plotly_chart(fig)