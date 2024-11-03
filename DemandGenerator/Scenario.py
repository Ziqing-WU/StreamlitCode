from config import *

with st.sidebar:
    """
    Navigation

    [Scenario Analysis](#scenario-analysis)
    - [Select the Scenarios](#select-the-scenarios)
    - [Apply Bass Model](#apply-bass-model)

    """
st.markdown(
    '''
    # Scenario Analysis
    Please create a folder called "scenario" and subfolders for each scenario, then put the corresponding data in the subfolders.
    '''
)
scenarios = st.multiselect("Select the scenarios to be analyzed", 