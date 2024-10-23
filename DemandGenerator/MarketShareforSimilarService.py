from email.mime import base
import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import geopandas as gpd
import pyproj
import plotly.express as px
import matplotlib as mpl
import matplotlib.pyplot as plt
import graphviz
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


st.set_page_config(
    layout='wide',
    initial_sidebar_state='auto',
    page_icon='🌒'
)

st.markdown(
    '''
    # Market Share for Similar Service
    ## Introduction
    The executive factors presented in the Home page can be quantified by statistics on different geographical or vehicle type level.
    The level of granularity depends on the availability of the data.
    ''')

with st.expander("Show status of data collection for each executive factor"):

    st.markdown(
    '''
|             | Title                                     | Complementary Information                                      | Statistics                                                                                                            | Granularity | Year       | Link                                                                                                                                                                                                    | Status             | Dependency |
|-------------|-------------------------------------------|----------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|-------------|------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------|-|
| D-C-1       | GDP/Capita                                |                                                                | PIB par habitant (euros)                                                                                              | Région      | 2020       | https://www.insee.fr/fr/statistiques/2012723#tableau-TCR_062_tab1_regions2016                                                                                                                           | :white_check_mark: | geo |
| D-EPTE-1    | Population Density                        |                                                                | Grille communale de densité                                                                                           | Commune     | 2022       | https://www.insee.fr/fr/information/6439600                                                                                                                                                             | :white_check_mark: | geo |
| ~~D-EPTE-2~~    | ~~Number of Households with Internet Access~~ |                                                                | ~~Ménages ayant accès à l'internet~~                                                                                      | ~~Pays~~        | ~~2021~~       | ~~https://www.insee.fr/fr/statistiques/2385835~~                                                                                                                                                            | ~~:question:~~ |    ~~geo~~ |
| Env-P-1     | Emission Reduction                        | Well-to-wheel GHG emission reduction comparing to alternatives |                                                                                                                       |             |            |                                                                                                                                                                                                         | :question:         | model |
| Env-P-2     | Average Temperature in Summer             | which may alter lifespan of electric vehicle                   | Maximum de moyenne de température quotidienne                                                                         | Département | 2021       | https://odre.opendatasoft.com/explore/dataset/temperature-quotidienne-departementale/information/?disjunctive.departement&sort=date_obs                                                                 | :white_check_mark: | geo |
| Env-P-3     | Average Temperature in Winter             | which may alter performance on range                           | Minimum de moyenne de température quotidienne                                                                         | Département | 2021       | https://odre.opendatasoft.com/explore/dataset/temperature-quotidienne-departementale/information/?disjunctive.departement&sort=date_obs                                                                 | :white_check_mark: | geo |
| Env-C-1     | Attitude towards Climate Change           |                                                                | Emission de CO2 par les navettes domicile/travail et domicile/étude, par km parcouru et personne au lieu de résidence | Département | 2011       | https://www.insee.fr/fr/statistiques/4505239 i095c lr                                                                                                                                                   | :white_check_mark: | geo |
| Env-EPTE-1  | ZFE-m                                     |                                                                | Situation ZFE-m (existante + à venir)                                                                                 | Commune     | 2022       | https://www.data.gouv.fr/fr/datasets/base-nationale-consolidee-des-zones-a-faibles-emissions/  https://www.service-public.fr/particuliers/actualites/A14587                                             | :clock1:           | geo (+ model) |
| E-P-1       | Acquisition Fee Reduction                 | Comparing to alternatifs                                       |                                                                                                                       |             |            |                                                                                                                                                                                                         | :question:         | model |
| E-P-2       | Garage Accessibility                      | which influences service accessibililty                        |                                                                                                                       |             | 2022       | https://www.auto-selection.com/garage-voiture/                                                                                                                                                          | :clock1:           | geo |
| E-C-1       | Household Disposable Income               |                                                                | Revenu fiscal déclaré médian par unité de consommation                                                                | Département | 2018       | https://www.insee.fr/fr/statistiques/4505239 i072                                                                                                                                                       | :white_check_mark: | geo |
| E-EPTE-1    | Charging Point Accessibility              |                                                                | Nombre de bornes de recharge par commune                                                                              | Commune     | 2022       | https://transport.data.gouv.fr/datasets/fichier-consolide-des-bornes-de-recharge-pour-vehicules-electriques/                                                                                            | :white_check_mark: | geo |
| ~~E-EPTE-2~~    | ~~Access to E-Lanes~~                         |                                                                |                                                                                                                       |             |            | ~~https://voyage.aprr.fr/articles/une-voie-reservee-aux-covoitureurs-taxis-et-vehicules-propres~~                                                                                                           | ~~:question:~~         | ~~geo~~ |
| E-EPTE-3    | Fuel Cost Saving vs. Gasoline             |                                                                |                                                                                                                       |             | real-time  | https://www.prix-carburants.gouv.fr/                                                                                                                                                                    | :clock1:           | geo |
| E-EPTE-4    | Fuel Cost Saving vs. Diesel               |                                                                |                                                                                                                       |             | real-time  | https://www.prix-carburants.gouv.fr/                                                                                                                                                                    | :clock1:           | geo |
| E-EPTE-5    | Financial Incentives                      |                                                                |                                                                                                                       |             |            | https://www.phoenixmobility.co/2020/10/19/subventions-retrofit-mode-emploi/                                                                                                                             | :clock1:           | geo |
| E-EPTE-6    | Existing Demo-Projects                    |                                                                |                                                                                                                       |             |            |                                                                                                                                                                                                         | :question:          | geo |
| Ener-EPTE-1 | Sustainable Electricity Production        | Share of sustainable electricity in net electricity generation | Part de la production d'électricité renouvelable dans la consommation totale d'électricité                            | Région      | 2018       | https://www.insee.fr/fr/statistiques/4505239 i038                                                                                                                                                       | :white_check_mark: | geo |
| T-P-1       | Commuting Behavior                        |                                                                | Distance médiane des navettes domicile-Travail pour les actifs occupés, pour les navetteurs (en km)*                   | Commune     | 2018       | https://www.insee.fr/fr/statistiques/4505239 i061b act                                                                                                                                                  | :white_check_mark: | geo |
| T-EPTE-1    | Car Density                               | number of cars / 1000 persons                                  |                                                                                                                       | Commune     | 2021, 2019 | parc auto: https://www.statistiques.developpement-durable.gouv.fr/donnees-sur-le-parc-automobile-francais-au-1er-janvier-2021 population: https://www.insee.fr/fr/statistiques/6011070?sommaire=6011075 | :clock1:           | geo |
 
 \* : Distance médiane de déplacement est la distance de déplacement entre le domicile et le lieu de travail parcourue pour les 50 % des déplacements domicile/travail les plus courts. La distance est calculée pour chaque individu comme la distance parcourue en automobile pour se rendre de sa commune de résidence à la commune où il travaille.
    '''
)
st.markdown(
    '''
    ## Workflow
    '''
)

with st.expander("Show graphical representation of workflow"):
    pro = graphviz.Digraph()
    pro.graph_attr['rankdir'] = 'TD' 
    pro.graph_attr['layout'] = 'dot'
    pro.attr('node', shape='box')
    pro.node('M1', label='Separate executive factors (EF) into 2 dimensions: geographical dependent \& vehicle type dependent')
    pro.node('M21', label='Define the granularity for geographical dependent EFs: Commune')
    pro.node('M22', label="Define the granularity for vehicle type dependent EFs: TVV / CNIT")
    pro.node('M31', label='Cluster')
    pro.node('M32', label='Cluster')
    pro.node('M41', label='Rank clusters')
    pro.node('M42', label='Rank clusters')
    pro.node('M51', label='Estimate adoption rate for each cluster')
    pro.node('M52', label='Estimate adoption rate for each cluster')
    pro.node('M6', label='Use 2 adoption rates as adoption probability for each vehicle in serviceable available market')

    pro.edge('M1', 'M21')
    pro.edge('M1', 'M22')
    pro.edge('M21', 'M31')
    pro.edge('M22', 'M32')
    pro.edge('M31', 'M41')
    pro.edge('M32', 'M42')
    pro.edge('M41', 'M51')
    pro.edge('M42', 'M52')
    pro.edge('M51', 'M6')
    pro.edge('M52', 'M6')


    st.graphviz_chart(pro)

@st.cache
def create_df_features(folder_path = r"C:\Users\zwu\OneDrive - IMT Mines Albi\Documents\Data\DemandForecasting"):
    """
    Convert metadata and data to pandas dataframe

    Parameters
    ----------
    folder_path : string
        The path containing all the cleaned data files (.csv)

    Returns 
    ----------
    df_features : a dataframe of metadata, indicating code, name, granularity, status of each feature
    dict_df: a dictionary of dataframes containing all features, where keys are geographical codes and values are feature values.
    """

    d = {'Num': ['D-C-1', 'D-EPTE-1', 'D-EPTE-2', 'Env-P-1', 'Env-P-2', 'Env-P-3', 'Env-C-1', 'Env-EPTE-1', 'E-P-1', 'E-P-2', 'E-C-1', 'E-EPTE-1', 'E-EPTE-2', 'E-EPTE-3', 'E-EPTE-4', 'E-EPTE-5', 'E-EPTE-6', 'Ener-EPTE-1', 'T-P-1', 'T-EPTE-1'],
    'Name': ['GDP/Capita', 'Population Density', 'Number of Households with Internet Access', 'Emission Reduction', 'Average Temperature in Summer', 'Average Temperature in Winter', 'Attitude towards Climate Change', 'ZFE-m', 'Acquisition Fee Reduction', 'Garage Accessibility', 'Household Disposable Income', 'Charging Point Accessibility', 'Access to E-Lanes', 'Fuel Cost Saving vs. Gasoline', 'Fuel Cost Saving vs. Diesel', 'Financial Incentives', 'Existing Demo-Projects', 'Sustainable Electricity Production', 'Commuting Behavior', 'Car Density'],
    'Level': ['REG', 'COM', 'PAYS', '', 'DEP', 'DEP', 'DEP', 'COM', '', '', 'DEP', 'COM', '', '', '', '', '', 'REG', 'COM', 'COM'],
    'Status': ['Ready', 'Ready', 'Pending', 'Pending', 'Ready', 'Ready', 'Ready', 'In Progress', 'Pending', 'In Progress', 'Ready', 'Ready', 'Pending', 'In Progress', 'In Progress', 'In Progress', 'Pending', 'Ready', 'Ready', 'In Progress']}
    df_features = pd.DataFrame(d)
    df_features_ready = df_features[df_features['Status']=='Ready']
    dict_df = {}
    for num in df_features_ready.Num:
        if num not in ['Env-P-2', 'Env-P-3']:
            path = os.path.join(folder_path+"\\"+num+'*'+"\\" + num + '-*.csv')
            file = glob.glob(path)
            df = pd.read_csv(file[0])
            dict_df[num] = df
        else:
            path = os.path.join(folder_path+"\\"+'Env-P-23'+'*'+"\\" + 'Env-P-23' + '-*.csv')
            file = glob.glob(path)
            df = pd.read_csv(file[0])
            if num == 'Env-P-2':
                dict_df[num] = df.iloc[:, :2]
            else:
                dict_df[num] = df.iloc[:, [0,2]]
    # change column name
    dict_df['D-C-1'].rename(columns={'PIB par habitant (euros) 2020':'D-C-1'}, inplace=True)
    dict_df['D-EPTE-1'].rename(columns={'Typo degr� densit�': 'D-EPTE-1'}, inplace=True)
    dict_df['Env-P-2'].rename(columns={'code_insee_departement':'DEP', 'Temp_summer': 'Env-P-2'}, inplace=True)
    dict_df['Env-P-3'].rename(columns={'code_insee_departement':'DEP', 'Temp_winter': 'Env-P-3'}, inplace=True)
    dict_df['Env-C-1'].rename(columns={'codgeo':'DEP', 'emi_CO2_kmpers_2011': 'Env-C-1'}, inplace=True)
    dict_df['E-C-1'].rename(columns={'codgeo':'DEP', 'revenu_decl_median_uc_2018': 'E-C-1'}, inplace=True)
    dict_df['E-EPTE-1'].rename(columns={'code_insee_commune': 'COM', 'Nombre_EV_charge_2022': 'E-EPTE-1'}, inplace=True)
    dict_df['Ener-EPTE-1'].rename(columns={'codgeo': 'REG', 'part_elec_renouv_2018': 'Ener-EPTE-1'}, inplace=True)
    dict_df['T-P-1'].rename(columns={'codgeo': 'COM', 'med_dist_navette_actif_2018': 'T-P-1'}, inplace=True)
    
    # change datatype for geographical code
    dict_df['D-C-1']['REG'] = dict_df['D-C-1']['REG'].astype(str).str.zfill(2)
    dict_df['Ener-EPTE-1'].REG = dict_df['Ener-EPTE-1'].REG.astype(str).str.zfill(2)

    # change datatype for indicators
    dict_df['D-C-1']['D-C-1'] = dict_df['D-C-1']['D-C-1'].str.replace(" ", "").astype('float64')

    # put geographical code as index
    for key in dict_df.keys():
        dict_df[key] = dict_df[key].set_index(df_features[df_features['Num']==key].Level.iloc[0])
    return df_features, dict_df


df_features, dict_df = create_df_features()

@st.cache
def create_gdf(level):
    """
    Create geopandas dataframe to construct maps

    Parameters
    ----------
    level : string
        Granularity of the map to be constructed

    Returns
    ----------
    geopandas dataframe containing geographical forms
    """
    folder_path = r'C:\Users\zwu\OneDrive - IMT Mines Albi\Documents\Data\DemandForecasting\Ref-1-CodeGeo\ADMIN-EXPRESS-COG_3-1__SHP__FRA_WM_2022-04-15\ADMIN-EXPRESS-COG\1_DONNEES_LIVRAISON_2022-04-15\ADECOG_3-1_SHP_WGS84G_FRA\\'
    if level=='COM':
        fp = folder_path+'COMMUNE.shp'
    elif level=='DEP': 
        fp = folder_path+'DEPARTEMENT.shp'
    elif level=='REG':
        fp = folder_path+'REGION.shp'
    map_df = gpd.read_file(fp)
    map_df.to_crs(pyproj.CRS.from_epsg(4326), inplace=True)
    return map_df[['INSEE_'+level, 'geometry']].set_index('INSEE_'+level)

def find_geo_names(level):
    """
    Create a dataframe to display the name of each geographical location
    """
    folder_path = r'C:\Users\zwu\OneDrive - IMT Mines Albi\Documents\Data\DemandForecasting\Ref-1-CodeGeo\\'
    if level=='COM':
        fp = folder_path+'commune_2022.csv'
        column = ['COM', 'NCC', 'DEP', 'REG']
    elif level=='DEP':
        fp = folder_path+'departement_2022.csv'
        column = ['DEP', 'NCC', 'REG']
    elif level=='REG':
        fp = folder_path+'region_2022.csv'
        column = ['REG', 'NCC']
    geo_names = pd.read_csv(fp)[column]
    if level in ['DEP', 'REG']:
        geo_names['REG'] = geo_names['REG'].astype(str).str.zfill(2)
    else:
        geo_names['REG'] = geo_names['REG'].astype(str).str.split(".", expand=True)[0].str.zfill(2)
    if level=='COM':
        geo_names.set_index('COM', inplace=True)
    if level=='DEP':
        geo_names.set_index('DEP', inplace=True)
    if level=='REG':
        geo_names.set_index('REG', inplace=True)
    return geo_names

tab1, tab2, tab3 = st.tabs(["City Clustering", "Vehicle Type Clustering", "Adoption Rate Application on SAM"])

with tab1:
    st.markdown(
        '''
        ### Cluster
        '''
    )
    with st.expander("Click here to see how clustering is done."):
        st.markdown(
            '''
            #### Visualization
            '''
        )
        indicator = st.selectbox('Select indicators to visualize', [i for i in dict_df.keys()])
        name = df_features[df_features['Num']==indicator].iloc[0,1]
        level = df_features[df_features['Num']==indicator].iloc[0,2]
        
        gdf = create_gdf(level)
        df_merged = gdf.merge(dict_df[indicator], left_index=True, right_index=True)

        @st.cache
        def draw_hist(indicator, name):
            hist = px.histogram(dict_df[indicator], x=indicator, title=name)
            return hist
        
        @st.cache
        def draw_map(indicator, level, name, df_merged):
            if (level in ['REG', 'DEP']) or indicator=='E-EPTE-1':
                fig_map = px.choropleth_mapbox(df_merged, geojson = df_merged['geometry'], color=indicator, locations=df_merged.index, mapbox_style="carto-positron", opacity=0.5, zoom=4, center = {"lat": 47, "lon": 2}, color_continuous_scale='Greys', title=name)
            else: 
                fig_map = None
            return fig_map

        if st.button('Visualize!'):
            hist = draw_hist(indicator, name)
            col1, col2 = st.columns([1,1])
            with col1:
                st.plotly_chart(hist)
            with col2:
                fig_map = draw_map(indicator, level, name, df_merged)
                if level in ['REG', 'DEP'] or indicator=='E-EPTE-1':
                    st.plotly_chart(fig_map)
                else:
                    fig, ax = plt.subplots(1, figsize=(10,10))
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="1%", pad=0.1)
                    df_merged.plot(column=indicator, cmap='Greys', linewidth=1, ax=ax, legend = True, cax=cax)
                    ax.axis('off')
                    st.pyplot(fig)
            dict_df[indicator]

        st.markdown(
                """
                #### Implication of business rules
                - T-P-1 Commuting Behavior: > 50 km

                Cities concerned:
                """
            )
        col1, col2 = st.columns([1,2])
        with col1:
            base_commune = find_geo_names('COM')
            base_commune.dropna(inplace=True)
            t_p_1 = base_commune.merge(dict_df['T-P-1'], how='left', left_index=True, right_index=True)
            delete_communes = t_p_1[t_p_1['T-P-1']>50]
            st.dataframe(delete_communes)

        with col2:
            gdf = create_gdf('COM')
            df_merged = gdf.merge(delete_communes, left_index=True, right_index=True)    
            fig_map = px.choropleth_mapbox(df_merged, geojson = df_merged['geometry'], color='T-P-1', locations=df_merged.index, mapbox_style="carto-positron", opacity=0.5, zoom=4, center = {"lat": 47, "lon": 2}, color_continuous_scale='Greys', hover_data=['NCC'])
            st.plotly_chart(fig_map)

        af_busi_base_commune = base_commune[~base_commune.index.isin(delete_communes.index)]

        def agg_df(dict_df, df_features, af_busi_base_commune):
            df = af_busi_base_commune
            for indicator in dict_df.keys():
                level = df_features[df_features['Num']==indicator].iloc[0,2]
                if level == 'COM':
                    df = df.merge(dict_df[indicator], how='left', left_index=True, right_index=True)
                elif level == 'DEP':
                    df = df.merge(dict_df[indicator], how='left', left_on='DEP', right_index=True)
                else:
                    df = df.merge(dict_df[indicator], how='left', left_on='REG', right_index=True)
            return df

        st.markdown(
            '''
            #### Check for empty values
            '''
            )
        
        agg_df = agg_df(dict_df, df_features, af_busi_base_commune)
        agg_df['E-EPTE-1'] = agg_df['E-EPTE-1'].fillna(0)
        columns = st.multiselect('Select features to be included', [i for i in dict_df.keys()], ['D-C-1', 'D-EPTE-1', 'Env-P-2', 'Env-P-3', 'Env-C-1', 'E-C-1', 'E-EPTE-1', 'T-P-1'])
        agg_df = agg_df[['NCC', 'DEP', 'REG']+columns]

        col1, col2 = st.columns([1,2])
        
        na_geo = agg_df[agg_df.isna().any(axis=1)]
        with col1:
            na_geo
        with col2:
            gdf = create_gdf('COM')
            df_merged = gdf.merge(na_geo, left_index=True, right_index=True)
            fig_map = px.choropleth_mapbox(df_merged, geojson = df_merged['geometry'], locations=df_merged.index, mapbox_style="carto-positron", opacity=0.5, zoom=4, center = {"lat": 47, "lon": 2}, color_continuous_scale='Greys', hover_data=['NCC'])
            st.plotly_chart(fig_map)

        st.markdown(
            '''
            Ener-EPTE-1 contains too much empty values, so it is dropped for the following analyses. Afterwards, lines with NaN values are also dropped.
            Dropped values here contain:
            - Extreme rural villages where the population is less than 10
            - The arrondissements of metropoles for which the whole metropole is already considered in the dataset
            - DROM-COM (départements et régions d'outre-mer et collectivités d'outre-mer)
            The first two categories are not interesting to be included into analysis. La France d'outre-mer will be analysed separately afterwards.
            '''
            )
        
        agg_df = agg_df[['NCC', 'DEP', 'REG', 'D-C-1', 'D-EPTE-1', 'Env-P-2', 'Env-P-3', 'Env-C-1', 'E-C-1', 'E-EPTE-1', 'T-P-1']]
        st.markdown(
            '''
            #### Scaling and Principal Component Analysis (PCA)
            Scaling needs to be done before conducting PCA because the variables don't have the same unit.
            Here [scaling](https://scikit-learn.org/stable/modules/preprocessing.html#standardization-or-mean-removal-and-variance-scaling) is explained. 
            Several scalers can be chosen in this case, including standard scaler, minmax scaler, maxabs scaler, and robust scaler.
            Then PCA is conducted on the scaled data.
            '''
            )
        
        nan_agg_df = agg_df.isna().sum(axis=1)
        
        # dropped cities
        nan_agg_df = nan_agg_df[nan_agg_df>0].index
        agg_df.dropna(inplace=True)
        agg_df_d = agg_df.iloc[:, 3:]

        scaler_type = st.selectbox(
            "Which scaler you like to choose?",
            ["Standard Scaler", "MinMax Scaler", "MaxAbs Scaler", "Robust Scaler"]
        )

        if scaler_type == "Standard Scaler":
            scaler = StandardScaler()
        elif scaler_type == "MinMax Scaler":
            scaler = MinMaxScaler()
        elif scaler_type == "MaxAbs Scaler":
            scaler = MaxAbsScaler()
        else:
            scaler = RobustScaler()

        scaled_agg_df_d = scaler.fit_transform(agg_df_d)

        pca = PCA()
        pca.fit(scaled_agg_df_d)
        
        col1, col2=st.columns([1,2])
        with col1:
            df = pd.DataFrame(pca.components_)
            df.index = ['PC'+str(i+1) for i in range(len(df.index))]
            df.columns = agg_df_d.columns
            df
            nb_dim=int(st.number_input("The number of dimensions to be considered", min_value=1, max_value=7, step=1, value=2, key=0))
            st.write("Explained variance ratio of the first ", nb_dim, " dimensions:", pca.explained_variance_ratio_[:nb_dim].sum())
        with col2:
            st.plotly_chart(px.bar(pca.explained_variance_ratio_, title='Scree Plot'))
        
        pca_2d = pca.transform(scaled_agg_df_d)
        pca_2d = pd.DataFrame(pca_2d[:, :2])
        pca_2d.index = agg_df_d.index
        pca_2d['NCC'] = agg_df['NCC']
        pca_2d['REG'] = agg_df['REG']
        pca_2d['dc1'] = agg_df['D-C-1']
        pca_2d.columns = ['PC1', 'PC2', 'NCC', 'REG', 'dc1']
        
        st.plotly_chart(
            px.scatter(pca_2d, x='PC1', y='PC2', hover_data=[pca_2d.index, 'NCC', 'REG'], opacity=0.5, color='REG', title='PCA per Region')
        )
        outliers_1st_pca = ['75056', '69123', '31555', '92044', '59350']

        outliers_1st_pca = st.multiselect("Select points to be eliminated for next PCA analysis", outliers_1st_pca, ['75056', '69123', '31555', '92044'])

        st.markdown(
            '''
            These cities will be analysed independently.
            '''
            )

        agg_df_outliers_1st_pca = agg_df[agg_df.index.isin(outliers_1st_pca)]
        agg_df = agg_df[~agg_df.index.isin(outliers_1st_pca)]
        
        agg_df_d = agg_df.iloc[:,3:]
        scaled_agg_df_d = scaler.fit_transform(agg_df_d)

        pca = PCA()
        pca.fit(scaled_agg_df_d)
        pca_agg_df_d = pca.transform(scaled_agg_df_d)
        col1, col2=st.columns([1,1])
        with col2:
            st.plotly_chart(px.bar(pca.explained_variance_ratio_, title='Scree Plot'))
        with col1:
            df = pd.DataFrame(pca.components_)
            df.index = ['PC'+str(i+1) for i in range(len(df.index))]
            df.columns = agg_df_d.columns
            df
            nb_dim=int(st.number_input("The number of dimensions to be considered", min_value=1, max_value=7, value=5, step=1, key=1))
            st.write("Explained variance ratio of the first ", nb_dim, " dimensions:", pca.explained_variance_ratio_[:nb_dim].sum())

        pca_nd = pd.DataFrame(pca.transform(pca_agg_df_d))
        pca_nd.columns = ['PC'+str(i) for i in range(1,9)]
        pca_nd.index = agg_df.index
        pca_2d_viz = pca_nd[['PC1', 'PC2']]
        pca_2d_viz['NCC'] = agg_df['NCC']
        pca_2d_viz['REG'] = agg_df['REG']
        st.plotly_chart(px.scatter(pca_2d_viz, x='PC1', y='PC2', hover_data=['NCC', 'REG'], color='REG', title='PCA per Region'))

        pca_nd = pca_nd[['PC'+str(i) for i in range(1,nb_dim+1)]]

        @st.cache
        def find_num_clusters(pca_2d):
            dict_score = {}
            for n in range(2,15):
                kmeans = KMeans(n_clusters = n, init='k-means++', random_state=0).fit(pca_2d)
                pca_2d["class"] = kmeans.labels_
                dict_score[n] = silhouette_score(pca_2d, labels=pca_2d["class"])
            return dict_score

        dict_score = find_num_clusters(pca_nd)
        st.write("The silhouette scores for each number of clusters are", dict_score)

        n=int(st.number_input("The number of clusters chosen is ", value=8))
        kmeans = KMeans(n_clusters = n, init='k-means++', random_state=0).fit(pca_nd)
        pca_nd["class"] = kmeans.labels_
        pca_nd['REG'] = agg_df['REG']
        pca_nd['NCC'] = agg_df['NCC']
        st.plotly_chart(px.scatter(pca_nd, x='PC1', y='PC2', hover_data=['NCC', 'REG'], color='class'))

        def convert_df(df):
            return df.to_csv().encode('utf-8')
            
        agg_df = agg_df.merge(pca_nd[['class']], left_index=True, right_index=True)
        csv = convert_df(agg_df)
        st.download_button("Download here", csv, mime='text/csv', file_name='clustering.csv')


        st.markdown(
            '''
            The medians of statistics for each group:
            '''
        )
        
        agg_df = agg_df.astype({"class": str})
        agg_df_groupby = pd.concat([agg_df.groupby(by = 'class').median(), agg_df_outliers_1st_pca.iloc[:, 3:]])
        st.dataframe(agg_df_groupby)

        st.markdown(
            """
            Check the statistics for each cluster
            """
        )
        feature = st.selectbox("Select feature to be visualized", agg_df.columns[3:-1])
        fig_box_bar = px.box(agg_df, x='class', y=feature, category_orders={'class': [str(i) for i in range(n)]})
        df = agg_df_outliers_1st_pca.iloc[:, 3:]
        fig_box_bar.add_box(y=df[feature], x=df.index, boxpoints='all')
        fig_box_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_box_bar)

        df_merged = gdf.merge(agg_df, left_index=True, right_index=True)

    fig, ax = plt.subplots(1, figsize=(10,10))
    divider = make_axes_locatable(ax)
    df_merged.plot(column='class', cmap='tab10', linewidth=1, ax=ax, legend = True)
    ax.axis('off')
    st.pyplot(fig)

    fig, ax = plt.subplots(1, figsize=(10,10))
    divider = make_axes_locatable(ax)
    df_merged.plot(linewidth=1, ax=ax, legend = True)
    ax.axis('off')
    st.pyplot(fig)    

    st.markdown(
        """
        ### Rank clusters
        """)

    def create_ahp_graph():
        ahp = graphviz.Graph()
        ahp.graph_attr['rankdir'] = 'LR' 
        ahp.graph_attr['layout'] = 'dot'
        ahp.attr('node', shape='box')
        ahp.node('goal', label="Which group of cities is good to deploy retrofit service?")
        
        ahp.node('c-demo-GDP', label='GDP/Capita')
        ahp.node('c-demo-pd', label='Population Density')
        ahp.node('c-env-summer', label='Average Temperature in Summer')
        ahp.node('c-env-winter', label='Average Temperature in Winter')
        ahp.node('c-env-climate', label='Attitude towards Climate Change')
        ahp.node('c-eco-income', label='Household Disposable Income')
        ahp.node('c-eco-charging', label='Charging Point Accessibility')
        ahp.node('c-trans-beh', label='Commuting Behavior')

        ahp.node('a-1', 'City Group 1')
        ahp.node('a-2', 'City Group 2')
        ahp.node('a-3', 'City Group 3')
        ahp.node('a-4', 'City Group 4')
        ahp.node('a-5', 'City Group 5')
        ahp.node('a-6', 'City Group 6')
        ahp.node('a-7', 'City Group 7')
        ahp.node('a-8', 'City Group 8')

        ahp.edge('goal', 'c-demo-GDP')
        ahp.edge('goal', 'c-demo-pd')

        ahp.edge('goal', 'c-env-summer')
        ahp.edge('goal', 'c-env-winter')
        ahp.edge('goal', 'c-env-climate')

        ahp.edge('goal', 'c-eco-income')
        ahp.edge('goal', 'c-eco-charging')
                
        ahp.edge('goal', 'c-trans-beh')

        for i in [str(i) for i in range(1,9)]:
            ahp.edge('c-demo-GDP', 'a-'+i)
            ahp.edge('c-demo-pd', 'a-'+i)

            ahp.edge('c-env-summer', 'a-'+i)
            ahp.edge('c-env-winter', 'a-'+i)
            ahp.edge('c-env-climate', 'a-'+i)

            ahp.edge('c-eco-income', 'a-'+i)
            ahp.edge('c-eco-charging', 'a-'+i)
                        
            ahp.edge('c-trans-beh', 'a-'+i)
        return ahp

    with st.expander("Click here to see how ranking of clusters is done"):
        st.write("Graphical representation of goal, criteria, alternatives")
        st.graphviz_chart(create_ahp_graph())
        st.write("The preference matrix of criteria is")

        column1, column2=st.columns([1.5,1])
        with column1:
            a=np.array([[1, 1/3, 1, 1/5, 1/7, 1/8, 1/4, 1/3],
                        [3, 1, 3, 1/3, 1/5, 1/6, 1/2, 1],
                        [1, 1/3, 1, 1/5, 1/7, 1/8, 1/4, 1/3],
                        [5, 3, 5, 1, 1/3, 1/4, 2, 3],
                        [7, 5, 7, 3, 1, 1/2, 4, 5],
                        [8, 6, 8, 4, 2, 1, 5, 6],
                        [4, 2, 4, 1/2, 1/4, 1/5, 1, 2],
                        [3, 1, 3, 1/3, 1/5, 1/6, 1/2, 1]])
            df_a = pd.DataFrame(a)
            df_a.index=["Env-P-2", "Env-P-3", "Env-C-1", "E-C-1", "E-EPTE-1", "T-P-1", "D-C-1", "D-EPTE-1"]
            df_a.columns=df_a.index
            df_a
        
        with column2:

            def ahp_get_weights(pref_matrix):
                n=len(pref_matrix)
                w=np.sum(pref_matrix, axis=0)
                norm=np.divide(pref_matrix, w)
                weights=np.sum(norm, axis=1)/n
                lambda_ = np.sum(np.divide(np.sum(weights*pref_matrix, axis=1), weights))/n
                CI = (lambda_-n)/(n-1)
                RI = {1:0, 2:0, 3:0.58, 4:0.9, 5:1.12, 6:1.24, 7:1.32, 8:1.41, 9:1.45, 10:1.49, 11:1.51, 12:1.48, 13:1.56, 14:1.57, 15:1.58}
                CR = CI/RI[n]
                if CR>0.1:
                    st.warning("Preference matrix used in AHP is inconsistent! CR is "+str(CR))
                return weights
        
            weights=ahp_get_weights(a)

            def construct_weights_df(weights, index):
                ahp_df = pd.DataFrame(weights)
                ahp_df.rename({0:"Weight"}, axis=1, inplace=True)
                ahp_df.index = index
                return ahp_df
            weights_df = construct_weights_df(weights, df_a.index)
            st.write("Weights for each criterion are", weights_df)
        
        st.write("Now let's evaluate the alternatives")
        d_c_1=[[1, 3, 1/3, 4, 1, 4, 1/5, 2, 2, 1/3, 1/6, 1/6],
            [1/3, 1, 1/4, 2, 1/3, 2, 1/6, 1/2, 1/2, 1/4, 1/8, 1/8],
            [3, 4, 1, 5, 3, 5, 1/4, 4, 4, 1/3, 1/5, 1/5],
            [1/4, 1/2, 1/5, 1, 1/4, 1, 1/7, 1/3, 1/3, 1/5, 1/9, 1/9],
            [1, 3, 1/3, 4, 1, 4, 1/5, 2, 2, 1/3, 1/6, 1/6], 
            [1/4, 1/2, 1/5, 1, 1/4, 1, 1/7, 1/3, 1/3, 1/5, 1/9, 1/9], 
            [5, 6, 4, 7, 5, 7, 1, 6, 6, 3, 1/3, 1/3], 
            [1/2, 2, 1/4, 3, 1/2, 3, 1/6, 1, 1, 1/4, 1/7, 1/7],
            [1/2, 2, 1/4, 3, 1/2, 3, 1/6, 1, 1, 1/4, 1/7, 1/7],
            [3, 4, 3, 5, 3, 5, 1/3, 4, 4, 1, 1/4, 1/4],
            [6, 8, 5, 9, 6, 9, 3, 7, 7, 4, 1, 1],
            [6, 8, 5, 9, 6, 9, 3, 7, 7, 4, 1, 1]
            ]
    
        d_epte_1=[[1, 3, 1/6, 2, 1, 2, 1/3, 1/4, 1/6, 1/6, 1/6, 1/6],
            [1/3, 1, 1/7, 1/2, 1/3, 1/2, 1/5, 1/6, 1/8, 1/8, 1/8, 1/8],
            [6, 7, 1, 7, 6, 7, 5, 3, 1/2, 1/2, 1/2, 1/2],
            [1/2, 2, 1/7, 1, 1/2, 1, 1/4, 1/5, 1/7, 1/7, 1/7, 1/7],
            [1, 3, 1/6, 2, 1, 2, 1/3, 1/4, 1/6, 1/6, 1/6, 1/6],
            [1/2, 2, 1/7, 1, 1/2, 1, 1/4, 1/5, 1/7, 1/7, 1/7, 1/7],
            [3, 5, 1/5, 4, 3, 4, 1, 1/3, 1/5, 1/5, 1/5, 1/5],
            [4, 6, 1/3, 5, 4, 5, 3, 1, 1/3, 1/3, 1/3, 1/3],
            [6, 8, 2, 7, 6, 7, 5, 3, 1, 1, 1, 1],
            [6, 8, 2, 7, 6, 7, 5, 3, 1, 1, 1, 1],
            [6, 8, 2, 7, 6, 7, 5, 3, 1, 1, 1, 1],
            [6, 8, 2, 7, 6, 7, 5, 3, 1, 1, 1, 1]]
    
        env_p_2=[[1, 1/4, 1/3, 1/4, 4, 1/3, 1/5, 1/5, 5, 5, 3, 1/3],
            [4, 1, 3, 1, 6, 3, 1/2, 1/2, 7, 7, 5, 2],
            [3, 1/3, 1, 1/3, 5, 1, 1/4, 1/4, 6, 6, 4, 1/2],
            [4, 1, 3, 1, 6, 3, 1/2, 1/2, 7, 7, 5, 2],
            [1/4, 1/6, 1/5, 1/6, 1, 1/5, 1/7, 1/7, 2, 2, 1/3, 1/5],
            [3, 1/3, 1, 1/3, 5, 1, 1/4, 1/4, 6, 6, 4, 1/2],
            [5, 2, 4, 2, 7, 4, 1, 1, 8, 8, 6, 3],
            [5, 2, 4, 2, 7, 4, 1, 1, 8, 8, 6, 3],
            [1/5, 1/7, 1/6, 1/7, 1/2, 1/6, 1/8, 1/8, 1, 1, 1/4, 1/6],
            [1/5, 1/7, 1/6, 1/7, 1/2, 1/6, 1/8, 1/8, 1, 1, 1/4, 1/6],
            [1/3, 1/5, 1/4, 1/5, 3, 1/4, 1/6, 1/6, 4, 4, 1, 1/4],
            [3, 1/2, 2, 1/2, 5, 2, 1/3, 1/3, 6, 6, 4, 1]]

        env_p_3=[[1, 1/2, 1/3, 2, 1/8, 2, 1, 1/4, 1/5, 1/3, 1/3, 1],
            [2, 1, 1/2, 3, 1/7, 3, 2, 1/3, 1/4, 1/2, 1/2, 2],
            [3, 2, 1, 4, 1/6, 4, 3, 1/2, 1/3, 1, 1, 3],
            [1/2, 1/3, 1/4, 1, 1/9, 1, 1/2, 1/5, 1/6, 1/4, 1/4, 1/2],
            [8, 7, 6, 9, 1, 9, 8, 5, 3, 6, 6, 8],
            [1/2, 1/3, 1/4, 1, 1/9, 1, 1/2, 1/5, 1/6, 1/4, 1/4, 1/2],
            [1, 1/2, 1/3, 2, 1/8, 2, 1, 1/4, 1/5, 1/3, 1/3, 1],
            [4, 3, 2, 5, 1/5, 5, 4, 1, 1/2, 2, 2, 4],
            [5, 4, 3, 6, 1/3, 6, 5, 2, 1, 3, 3, 5],
            [3, 2, 1, 4, 1/6, 4, 3, 1/2, 1/3, 1, 1, 3],
            [3, 2, 1, 4, 1/6, 4, 3, 1/2, 1/3, 1, 1, 3],
            [1, 1/2, 1/3, 2, 1/8, 2, 1, 1/4, 1/5, 1/3, 1/3, 1]
            ]

        env_c_1=[[1, 2, 1/2, 2, 2, 1, 1/5, 2, 1, 1/3, 1/8, 1/6],
            [1/2, 1, 1/3, 1, 1, 1/2, 1/6, 1, 1/2, 1/4, 1/9, 1/7],
            [2, 3, 1, 3, 3, 2, 1/4, 3, 2, 1/2, 1/7, 1/5],
            [1/2, 1, 1/3, 1, 1, 1/2, 1/6, 1, 1/2, 1/4, 1/9, 1/7],
            [1/2, 1, 1/3, 1, 1, 1/2, 1/6, 1, 1/2, 1/4, 1/9, 1/7],
            [1, 2, 1/2, 2, 2, 1, 1/5, 2, 1, 1/3, 1/8, 1/6],
            [5, 6, 4, 6, 6, 5, 1, 6, 5, 3, 1/5, 1/3],
            [1/2, 1, 1/3, 1, 1, 1/2, 1/6, 1, 1/2, 1/4, 1/9, 1/7],
            [1, 2, 1/2, 2, 2, 1, 1/5, 2, 1, 1/3, 1/8, 1/6],
            [3, 4, 2, 4, 4, 3, 1/3, 4, 3, 1, 1/6, 1/4],
            [8, 9, 7, 9, 9, 8, 5, 9, 8, 6, 1, 4],
            [6, 7, 5, 7, 7, 6, 3, 7, 6, 4, 1/4, 1]
            ]
    
        e_c_1=[[1, 1/5, 1/2, 1/5, 1/4, 1/2, 3, 1/3, 1, 1, 5, 5],
            [5, 1, 4, 1, 2, 4, 6, 3, 5, 5, 8, 8],
            [2, 1/4, 1, 1/4, 1/3, 1, 3, 1/2, 2, 2, 5, 5],
            [5, 1, 4, 1, 2, 4, 6, 3, 5, 5, 8, 8],
            [4, 1/2, 3, 1/2, 1, 3, 5, 2, 4, 4, 7, 7],
            [2, 1/4, 1, 1/4, 1/3, 1, 3, 1/2, 2, 2, 5, 5],
            [1/3, 1/6, 1/3, 1/6, 1/5, 1/3, 1, 1/4, 1/3, 1/3, 3, 3],
            [3, 1/3, 2, 1/3, 1/2, 2, 4, 1, 3, 3, 6, 6],
            [1, 1/5, 1/2, 1/5, 1/4, 1/2, 3, 1/3, 1, 1, 5, 5],
            [1, 1/5, 1/2, 1/5, 1/4, 1/2, 3, 1/3, 1, 1, 5, 5],
            [1/5, 1/8, 1/5, 1/8, 1/7, 1/5, 1/3, 1/6, 1/5, 1/5, 1, 1],
            [1/5, 1/8, 1/5, 1/8, 1/7, 1/5, 1/3, 1/6, 1/5, 1/5, 1, 1]]

        e_epte_1=[[1, 1, 1/4, 1, 1, 1, 1/2, 1/2, 1/6, 1/7, 1/9, 1/5],
            [1, 1, 1/4, 1, 1, 1, 1/2, 1/2, 1/6, 1/7, 1/9, 1/5],
            [4, 4, 1, 4, 4, 4, 3, 3, 1/4, 1/6, 1/8, 1/3],
            [1, 1, 1/4, 1, 1, 1, 1/2, 1/2, 1/6, 1/7, 1/9, 1/5],
            [1, 1, 1/4, 1, 1, 1, 1/2, 1/2, 1/6, 1/7, 1/9, 1/5],
            [1, 1, 1/4, 1, 1, 1, 1/2, 1/2, 1/6, 1/7, 1/9, 1/5],
            [2, 2, 1/3, 2, 2, 2, 1, 1/2, 1/6, 1/7, 1/9, 1/5],
            [2, 2, 1/3, 2, 2, 2, 1, 1/2, 1/6, 1/7, 1/9, 1/5],
            [6, 6, 4, 6, 6, 6, 6, 6, 1, 1/3, 1/6, 2],
            [7, 7, 6, 7, 7, 7, 7, 7, 3, 1, 1/5, 4],
            [9, 9, 8, 9, 9, 9, 9, 9, 6, 5, 1, 6],
            [5, 5, 3, 5, 5, 5, 5, 5, 1/2, 1/4, 1/6, 1]
    ]

        t_p_1=[[1, 1/4, 1/5, 1/2, 1/3, 1/3, 1/2, 1/4, 1/6, 1/6, 1/6, 1/5],
            [4, 1, 1/2, 3, 2, 2, 3, 1, 1/3, 1/3, 1/3, 1/2],
            [5, 2, 1, 4, 3, 3, 4, 2, 1/2, 1/2, 1/2, 1],
            [2, 1/3, 1/4, 1, 1/2, 1/2, 1, 1/3, 1/5, 1/5, 1/5, 1/4],
            [3, 1/2, 1/3, 2, 1, 1, 2, 1/2, 1/4, 1/4, 1/4, 1/3],
            [3, 1/2, 1/3, 2, 1, 1, 2, 1/2, 1/4, 1/4, 1/4, 1/3],
            [2, 1/3, 1/4, 1, 1/2, 1/2, 1, 1/3, 1/5, 1/5, 1/5, 1/4],
            [4, 1, 1/2, 3, 2, 2, 3, 1, 1/3, 1/3, 1/3, 1/2],
            [6, 3, 2, 5, 4, 4, 5, 3, 1, 1, 1, 2],
            [6, 3, 2, 5, 4, 4, 5, 3, 1, 1, 1, 2],
            [6, 3, 2, 5, 4, 4, 5, 3, 1, 1, 1, 2],
            [5, 2, 1, 4, 3, 3, 4, 2, 1/2, 1/2, 1/2, 1]
            ]

        dict_pref_alt = {"D-C-1":d_c_1, "D-EPTE-1":d_epte_1, "Env-P-2":env_p_2, "Env-P-3":env_p_3, "Env-C-1":env_c_1, "E-C-1":e_c_1, "E-EPTE-1":e_epte_1, "T-P-1":t_p_1}
        crit = st.selectbox("Check the preference matrix and weights of alternatives for a criterion", df_a.index)
        column1, column2 = st.columns([2,1])
        with column1:
            df_crit = pd.DataFrame(dict_pref_alt[crit])
            df_crit_column = [str(i) for i in range(8)] + ["31555", "69123", "75056", "92044"]
            df_crit.columns = df_crit_column
            df_crit.index = df_crit_column
            df_crit
        with column2:
            df_weights_crit_alt = ahp_get_weights(dict_pref_alt[crit])
            st.write(construct_weights_df(df_weights_crit_alt, df_crit_column))

        st.write(
            "Final scores for each alternative "
        )

        dfs = []
        for crit in df_a.index:
            df = construct_weights_df(ahp_get_weights(dict_pref_alt[crit]), df_crit_column)
            df.columns = ["Weights " + crit]
            dfs.append(df)
        df_all = pd.concat(dfs, axis=1)
        
        final_scores = (df_all*weights_df.values.T).sum(axis=1)
        final_scores.sort_values(ascending=False, inplace=True)
        final_scores
        
        agg_df_outliers_1st_pca['class']=agg_df_outliers_1st_pca.index
        agg_df=pd.concat([agg_df, agg_df_outliers_1st_pca], axis=0)
        my_order = final_scores.index

        my_order_dict = {key: i for i, key in enumerate(my_order)}
        agg_df['order']=agg_df['class'].apply(lambda d: my_order_dict[d])
        df_merged = gdf.merge(agg_df, left_index=True, right_index=True)
        df_merged.sort_values(by='order', inplace=True)
        # file = df_merged.to_file("chorepleth_data")
        # st.download_button("Download here", file, file_name='chorepleth_data.shp')


    fig, ax = plt.subplots(1, figsize=(10,10))
    divider = make_axes_locatable(ax)
    df_merged.plot(column='order', cmap='Oranges_r', linewidth=1, ax=ax, legend = False)
    ax.axis('off')
    st.pyplot(fig)

    st.markdown(
        """
        ### Estimate adoption rate for each cluster
        """
    )

    final_score_scaled = final_scores/final_scores.max()

    max_adoption_rate = st.number_input("The maximum adoption rate is ", value=0.3, min_value=0.0, max_value=1.0)
    adoption_rates = max_adoption_rate*final_score_scaled
    st.write(
        """
        Adoption rate for each cluster is then
        """, adoption_rates)
    adoption_rates.rename("adop_rate", inplace=True)

    csv = convert_df(adoption_rates)
    st.download_button("Download here", csv, mime='text/csv', file_name='adoption_rates.csv')

    geo_info = df_merged.merge(adoption_rates, how='left', left_on='class', right_index=True)[["class", "adop_rate"]]
    # geo_info

with tab3:
    current_year = st.radio('Show Market Situation in Year ', [2022, 2023, 2024, 2025, 2026], horizontal=True)
    csv_file = st.file_uploader("Upload all vehicles included in Serviceable Available Market here")
    if csv_file is not None:
        df_vehicle = pd.read_csv(csv_file)
    else:
        file_path = r"C:\Users\zwu\OneDrive - IMT Mines Albi\Documents\Data\DemandForecasting\Vehicles\Simulated_car_registration\\"+str(current_year)+"_vehicle_list_SAM.csv"
        df_vehicle = pd.read_csv(file_path).iloc[:,2:]
        df_vehicle = df_vehicle.astype({'Code_commune':str})
    
    df_vehicle = df_vehicle.merge(geo_info, how='left', left_on="Code_commune", right_index=True)
    df_vehicle['retrofit_service'] = np.nan
    for i in range(df_vehicle.shape[0]):
        proba_retrofit = df_vehicle.iloc[i]["adop_rate"]
        df_vehicle.iloc[i,-1] = np.random.choice([False,True], p=[1-proba_retrofit, proba_retrofit])
    
    st.write(df_vehicle)

    # def check_perf_matrix(a, rtol=1e-05, atol=1e-08):
    #     a = np.array(a)
    #     return np.multiply(a, a.T)