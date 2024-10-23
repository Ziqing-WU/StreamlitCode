import streamlit as st


st.set_page_config(
    layout='wide',
    initial_sidebar_state='auto'
)

def page1():
    st.write(st.session_state.foo)

pages = {
    "Demand": [
        st.Page("Intro.py", title="Home", icon='🏠'),
        st.Page("TotalMarket.py",title="Total Market", icon='🌕'),
        st.Page("TotalAddressableMarket.py", title="Total Addressable Market", icon='🌔'),
        st.Page("ServiceableAvailableMarket.py", title="Serviceable Addressable Market", icon='🌓'),
        st.Page("MarketShareforSimilarService.py", title="Demand Estimation for all companies targeting the same Market", icon='🌒'),
        st.Page("MarketSharefortheCompany.py", title="Demand Estimation for the Company", icon='🌛'),
        st.Page("Scenario.py", title="Scenario Analysis", icon='📊')
    ],
    "Supply": [
        st.Page(page1)
    ]
}

pg = st.navigation(pages)
pg.run()

