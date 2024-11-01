from config import *

with st.sidebar:
    """
    Navigation

    [Demand Estimation](#demand-estimation)
    - [Demand Estimation Framework](#demand-estimation-framework)
    - [Application on Electric Retrofit of ICE Vehicle](#application-on-electric-retrofit-of-ice-vehicle)
    """


st.markdown(
"""
# Demand Estimation 
Demand estimation model provides scenarios of demands for new circular service on each product model, at each city and during each period, serves as the input of Decision Support System for circular supply chain network design.
The outline of the whole PhD project is shown as below. The demand estimation model is one of the three main components presented here.
"""
)

st.image('BigPicture.png', width=800, caption='Overview of Project')

st.markdown(
    """
    ## Demand Estimation Framework
    Here is the proposed framework that will be used to estimate demand, which is composed of sets and filters.
    
    """
)

st.image('PoupeeRusse.png', width=800, caption='Overview of Demand Estimation Framework')

"""
## Application on Electric Retrofit of ICE Vehicle
"""
col1, col2 = st.columns([1,1])

with col1:
    
    """
    The analysis pipeline in the demand estimation section is illustrated here: 
    """
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

with col2:
    st.markdown(
        """
        The framework is applied to the case of electric retrofitting of internal combustion engine (ICE) vehicles, inspired by a start-up where I spent 8 months observing their operations. The filters are defined as follows:
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
        - Filter 3: Consumer Purchasing Intention Analysis
            - Executive Factors Identification (click the following expander for an example)
        - Filter 4: Competitiveness of the Company
    
        *N.B.*: This information is used to set default values. Users are welcome to change these filters based on their needs.
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
    
    inspired from [Zubaryeva et al., 2012](https://doi.org/10.1016/j.techfore.2012.06.004)
        """
    )
"""
Get ready to explore the exciting world of demand estimation for circular businesses—click the button here and join the journey!
"""

if st.button("Start with the Total Market!"):
    st.switch_page("TotalMarket.py")