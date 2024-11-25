from config import *

st.title('Network Design')

st.write("""
The network design proposes the strategic decisions related to the location of facilities. 
It takes estimated demand in the previous step into account.
Only the default scenarios are considered in this step, meanining:
- Demand Potential: with TVVs covering 10%, 20%, 30% of the market
- Demand Dynamics: with baseline, word-of-mouth dominant, and advertising dominant scenarios (various values of p and q in the Bass Model)
- Collaborative Strategies: integrated, together, and hyperconnected
""")

st.image("DSS-NetworkDesign-Overview.png", caption="Overview of the Network Design: Scenarios and Outputs")
st.write("The network topology is shown in the following graph. The nodes represent the facilities, and the edges represent the flows between the facilities. The part in gray represents the external entities, which are not considered in the network design.")
st.graphviz_chart(
    """
digraph Network {
    node [shape=box]; // Sets the shape of nodes to be boxes

    // Define node styles
    "Factory (F)" ;
    "Logistics Node (L)" ;
    "Retrofit Center (R)";
    "Market Segment (M)";
    "Recovery Center (V)" ;
    supplier [shape=ellipse color="gray70" fontcolor="gray70"];
    "recycling center" [shape=ellipse color="gray70" fontcolor="gray70"];
    "EoL Vehicle processing center" [shape=ellipse color="gray70" fontcolor="gray70"];
   

    // Define same rank (horizontal alignment)
    //{ rank=same; "Factory (F)" , "Logistics Node (L)", "Retrofit Center (R)", supplier  }

    // Define edges
    supplier -> "Factory (F)" [color="gray70" headlabel="components" labelfontcolor="gray70"];
    "Factory (F)" -> "Logistics Node (L)" [label="RU"];
    "Logistics Node (L)" -> "Retrofit Center (R)" [label="RU"];
    "Logistics Node (L)" -> "Logistics Node (L)" [label="RU"];
    "Market Segment (M)" -> "Retrofit Center (R)" [label="EoLP"];
    "Retrofit Center (R)" -> "Market Segment (M)" [label="RP"];
    //"Retrofit Center (R)" -> "Recovery Center (V)" [label="ExPa"];
    "Retrofit Center (R)" -> "EoL Vehicle processing center"[color="gray70" headlabel="retrieved parts & EoL vehicles" labelfontcolor="grey70"];
    "Recovery Center (V)" -> "Logistics Node (L)" [label="RU"];
    "Recovery Center (V)" -> "Factory (F)" [label="EoLPa"];
    "Market Segment (M)" -> "Retrofit Center (R)" [label="PR"];
    "Retrofit Center (R)" -> "Recovery Center (V)" [label="EoLRU"];
    "Recovery Center (V)" -> "recycling center"[color="gray70" headlabel="recyclable materials" labelfontcolor="grey70"];
	//"Recovery Center (V)" -> "recycling center" [label="qe_mrpt-q_vfpt-q_vlpt"];
}
""")

st.write("The following page will show the data input for the network design models.")

