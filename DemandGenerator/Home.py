import streamlit as st


st.set_page_config(
    layout='wide',
    initial_sidebar_state='auto'
)



pages = {
    "Demand": [
        st.Page("Intro.py", title="Introduction", icon='🏠'),
        st.Page("TotalMarket.py",title="Total Market", icon='🌕'),
        st.Page("TotalAddressableMarket.py", title="Total Addressable Market", icon='🌔'),
        st.Page("ServiceableAvailableMarket.py", title="Serviceable Addressable Market", icon='🌓'),
        st.Page("MarketShareforSimilarService.py", title="Demand Estimation for all companies targeting the same Market", icon='🌒'),
        st.Page("MarketSharefortheCompany.py", title="Demand Estimation for the Company", icon='🌛'),
        st.Page("Scenario.py", title="Scenario Analysis", icon='📊')
    ],
    "Supply": [
        st.Page("IntroSupply.py", title="Introduction", icon='🏠'),
        st.Page("DataPrep.py", title="Data Input", icon='🔢'),
    ]
}

pg = st.navigation(pages)
pg.run()

