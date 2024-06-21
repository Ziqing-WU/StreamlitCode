import streamlit as st
import pandas as pd


# Create a main page title
st.title('Hyperconnected Circular Supply Chain Network Design')

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
