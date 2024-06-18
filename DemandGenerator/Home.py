import streamlit as st
import graphviz

st.set_page_config(
    layout='wide',
    initial_sidebar_state='auto'
)

st.markdown(
"""
# Demand Generator
Demand generator provides scenarios of demands for retrofit service on each product, at each city and during each period, serves as the input of Supply Chain Designer.
The outline of this project is described as below. The demand generator is one of the three main components presented here.
"""
)

st.image('Framework.png', width=800, caption='Overview of Project')

st.markdown(
    """
    ## Components of Demand Generator
    An overview of the demand generator structure is shown as following.
    """
)

st.image('PoupeeRusse.png', width=800, caption='Overview of Demand Generator')

col1, col2 = st.columns([1,1])

with col2:
    pro = graphviz.Digraph()
    pro.attr('node', shape='box')
    pro.node('M1', label='Total Market')
    pro.node('M2', label="Total Addressable Market")
    pro.node('M3', label='Serviceable Available Market')
    pro.node('M4', label='Market Share for Similar Service')
    pro.node('M5', label='Market Share for the Company')

    pro.edge('M1', 'M2', label='Filter 1: Legislative and Technical Constraints')
    pro.edge('M2', 'M3', label='Filter 2: Strategies and Business Model of the Company')
    pro.edge('M3', 'M4', label='Filter 3: Consumer Purchasing Intention Analysis')
    pro.edge('M4', 'M5', label='Filter 4: Competitiveness of the Company')

    st.graphviz_chart(pro)

with col1:
    st.markdown(
        """
        - Filter 1: Legislative and Technical Constraints
            - Legislative Constraints
                - Vehicle Category: L, M, N, not registered as collection vehicle
                - Vehicle Age: more than 3 years for category L, more than 5 years for category M, N
                - Vehicle Engine: at least one spark-ignition or compression engine
                - Geographical Coverage: registered in France
            - Technical Constraints
                - Vehicle Type : M1, N1
                - Vehicle Engine: diesel / petrol
                - Engine Power: between 60 kW and 110 kW
                - and other constraints on weights, noise, CO2 emission, etc
        - Filter 2: Strategies and Business Model of the Company
            - Geographical Coverage: urban area
            - Vehicle Model Selection: last vehicle of the model is fabricated at least 5 years ago / models with most vehicles
            - Consumer Type: toB / toC
        - Filter 3: Consumer Decision Making
            - Executive Factors Identification 
                - toB
                - toC (see table as following for more details)
        - Filter 4: Competitiveness of the Company
        """
    )
with st.expander('Executive Factors for Consumers'):

    st.markdown(
        """
    |            | From existing product (P)                                                                                                                                           | From consumer (**C**)                  | From economical, physical and technological environment (**EPTE**)                                                                                                |
    |-------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|
    | Demography (**D**)  |                                                                                                                                                                 | GDP/capita (**D-C-1**)                     | Population density (**D-EPTE-1**); number of households with internet access (**D-EPTE-2**)                                                                                          |
    | Environment (**Env**)| Well-to-wheel GHG emission reduction comparing to alternatives (**Env-P-1**); average temperature in summer (lifespan damage) (**Env-P-2**); average temperature in winter (range reduction) (**Env-P-3**) | Attitude towards climate change (**Env-C-1**) | ZFE-m (**Env-EPTE-1**)                                                                                                                                                  |
    | Economy (**E**)    | Acquisition fee reduction comparing to alternatives (**E-P-1**); garage accessibility (**E-P-2**)                                                                                       | Household disposable income (**E-C-1**)     | Charging point accessibility (**E-EPTE-1**); access to e-lanes (**E-EPTE-2**); fuel cost saving vs gasoline (**E-EPTE-3**); fuel cost saving vs diesel (**E-EPTE-4**); financial incentives (**E-EPTE-5**); existing demo-projects (**E-EPTE-6**) |
    | Energy (**Ener**)     |                                                                                                                                                                |                                | Share of sustainable electricity in net electricity generation (**Ener-EPTE-1**)                                                                                          |
    | Transport (**T**)  | Commuting behavior (considering the limits of range) (**T-P-1**)                                                                                                            |                                | Car density (cars/1000 habitants) (**T-EPTE-1**)                                                                                                                      |

        """
    )

