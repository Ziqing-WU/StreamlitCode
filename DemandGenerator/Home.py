import streamlit as st


st.set_page_config(
    layout='wide',
    initial_sidebar_state='auto'
)



pages = {
    "Demand": [
        st.Page("Demand1-Intro.py", title="Introduction", icon='🏠'),
        st.Page("Demand2-TotalMarket.py",title="Total Market", icon='🌕'),
        st.Page("Demand3-TotalAddressableMarket.py", title="Total Addressable Market", icon='🌔'),
        st.Page("Demand4-ServiceableAvailableMarket.py", title="Serviceable Addressable Market", icon='🌓'),
        st.Page("Demand5-MarketShareforSimilarService.py", title="Demand Estimation for all companies targeting the same Market", icon='🌒'),
        st.Page("Demand6-MarketSharefortheCompany.py", title="Demand Estimation for the Company", icon='🌛'),
        st.Page("Demand7-Scenario.py", title="Scenario Analysis", icon='📊')
    ],
    "Supply": [
        st.Page("Supply1-IntroSupply.py", title="Introduction", icon='🏠'),
        st.Page("Supply2-DataPrep.py", title="Data Input", icon='🔢'),
        st.Page("Supply3-Model.py", title="Optimisation Model", icon='🧮'),
        st.Page("Supply4-Result.py", title="Result Analysis", icon='🔍')
    ]
}

pg = st.navigation(pages)
pg.run()

