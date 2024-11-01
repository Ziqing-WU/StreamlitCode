from config import *

with st.sidebar:
    """
    Navigation

    [Demand Estimation for the Market](#demand-estimation-for-the-market)
    - [Executive Factors](#executive-factors)
    - [Workflow](#workflow)
    """

@st.cache_resource
def find_num_clusters(pca_2d):
    dict_score = {}
    for n in range(2,25):
        kmeans = KMeans(n_clusters = n, init='k-means++', random_state=0).fit(pca_2d)
        dict_score[n] = silhouette_score(pca_2d, labels=kmeans.labels_)
    return dict_score

st.markdown(
    '''
    # Demand Estimation for the Market
    ## Executive Factors
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

@st.cache_data
def create_df_features(folder_path = executive_factor_folder):
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
            df = pd.read_csv(file[0], encoding="utf-8", sep=",", encoding_errors="replace")
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

@st.cache_data
def create_gdf(level, is_occitanie=False):
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
    folder_path = rf'{executive_factor_folder}\Ref-1-CodeGeo\ADMIN-EXPRESS-COG_3-1__SHP__FRA_WM_2022-04-15\ADMIN-EXPRESS-COG\1_DONNEES_LIVRAISON_2022-04-15\ADECOG_3-1_SHP_WGS84G_FRA\\'
    suffix=""
    if is_occitanie:
        folder_path = rf'{executive_factor_folder}\Ref-1-CodeGeo\\'
        suffix="_OCCITANIE"
    if level=='COM':
        fp = folder_path+'COMMUNE'+suffix+'.shp'
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
    folder_path = rf'{executive_factor_folder}\Ref-1-CodeGeo\\'
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

tab1, tab2, tab3 = st.tabs(["City Clustering", "Vehicle Model Clustering", "Adoption Rate Application on SAM"])

with tab1:
    st.markdown(
        '''
        ### Cluster
        '''
    )
    with st.expander("Click here to see how clustering is done on cities"):
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

        @st.cache_data
        def draw_hist(indicator, name):
            hist = px.histogram(dict_df[indicator], x=indicator, title=name)
            return hist
        
        @st.cache_resource
        def draw_map(indicator, level, name, _df_merged):
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
        columns = st.multiselect('Select features to be included in the following display', [i for i in dict_df.keys()], ['D-C-1', 'D-EPTE-1', 'Env-P-2', 'Env-P-3', 'Env-C-1', 'E-C-1', 'E-EPTE-1', 'T-P-1'])
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
            Scaling needs to be done before conducting dimension reduction with PCA because the variables don't have the same unit.
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
        
        agg_df_d_melted = agg_df_d.reset_index().melt(id_vars='COM', var_name='category', value_name='value')
        # st.write(agg_df_d_melted.head())
        # fig = px.box(agg_df_d_melted, x='category', y='value')
        # st.plotly_chart(fig)


        scaler_type = st.selectbox(
            "Which scaler would you like to choose?",
            ["Standard Scaler", "MinMax Scaler", "MaxAbs Scaler", "Robust Scaler"],
            index=3
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
            nb_dim=int(st.number_input("The number of dimensions to be considered", min_value=1, max_value=7, value=2, step=1, key=1))
            st.write("Explained variance ratio of the first ", nb_dim, " dimensions:", pca.explained_variance_ratio_[:nb_dim].sum())

        pca_nd = pd.DataFrame(pca_agg_df_d)
        pca_nd.columns = ['PC'+str(i) for i in range(1,9)]
        pca_nd.index = agg_df.index
        pca_2d_viz = pca_nd[['PC1', 'PC2']]
        pca_2d_viz.loc[:,'NCC'] = agg_df['NCC']
        pca_2d_viz.loc[:,'REG'] = agg_df['REG']
        st.plotly_chart(px.scatter(pca_2d_viz, x='PC1', y='PC2', hover_data=['NCC', 'REG'], color='REG', title='PCA per Region'))

        pca_nd = pca_nd[['PC'+str(i) for i in range(1,nb_dim+1)]]

        dict_score = find_num_clusters(pca_nd)
        st.write("The silhouette scores for each number of clusters are", dict_score)

        n=int(st.number_input("The number of clusters chosen is ", value=7))
        kmeans = KMeans(n_clusters = n, init='k-means++', random_state=0).fit(pca_nd)
        pca_nd["class"] = kmeans.labels_
        pca_nd['REG'] = agg_df['REG']
        pca_nd['NCC'] = agg_df['NCC']
        st.plotly_chart(px.scatter(pca_nd, x='PC1', y='PC2', hover_data=['NCC', 'REG'], color='class'))

        def convert_df(df):
            return df.to_csv().encode('utf-8')
            
        agg_df = agg_df.merge(pca_nd[['class']], left_index=True, right_index=True)
        csv = convert_df(agg_df)
        st.download_button("Download the clustering results here", csv, mime='text/csv', file_name='clustering.csv')


        st.markdown(
            '''
            The medians of statistics for each group:
            '''
        )
        
        agg_df = agg_df.astype({"class": str})
        agg_df_groupby = pd.concat([agg_df.groupby(by = 'class').median(numeric_only=True), agg_df_outliers_1st_pca.iloc[:, 3:]])
        st.dataframe(agg_df_groupby)

        st.markdown(
            """
            Check the statistics for each cluster
            """
        )
        feature = st.selectbox("Select feature to be visualized", agg_df.columns[3:-1])
        df = agg_df_outliers_1st_pca.iloc[:, 3:]
        df.reset_index(inplace=True)
        df["COM"] = df["COM"].astype(str)
        x_categories = [str(i) for i in range(n)] + df["COM"].tolist()
        fig = go.Figure()
        for i in range(n):
            fig.add_trace(go.Box(
                y=agg_df[agg_df['class'] == str(i)][feature],
                x=[str(i)]*len(agg_df[agg_df['class'] == str(i)]),
                name=f'Class {str(i)}'
            ))
        
        fig.add_trace(go.Scatter(
            x=df['COM'],
            y=df[feature],
            mode='markers',
            marker=dict(color='red', size=6, symbol='circle'),
            name='Outliers',
            text=df['COM'],  # Add 'code commune' as hover text
            hovertemplate='Code Commune: %{text}<br>Feature Value: %{y}'
        ))
        fig.update_layout(
            xaxis=dict(
                title='Class / Code Commune',
                type='category',
                categoryorder='array',
                categoryarray=x_categories
            ),
            yaxis_title=feature,
            showlegend=False
        )
        st.plotly_chart(fig)
        cluster_stats = agg_df.groupby('class')[feature].agg(
            median='median',
            Q1=lambda x: x.quantile(0.25),
            Q3=lambda x: x.quantile(0.75),
            min='min',
            max='max'
        )
        city_stats = df.groupby('COM')[feature].agg(
            median='median',
            Q1=lambda x: x.quantile(0.25),
            Q3=lambda x: x.quantile(0.75),
            min='min',
            max='max'
        )
        combined_stats = pd.concat([cluster_stats, city_stats])
        st.write(combined_stats)


    if st.checkbox("Show only Occitanie Region", value=True):
        gdf = create_gdf("COM", is_occitanie=True)  
        df_merged = gdf.merge(agg_df, left_index=True, right_index=True)
    else:
        gdf = create_gdf("COM")
        df_merged = gdf.merge(agg_df, left_index=True, right_index=True)

    fig, ax = plt.subplots(1, figsize=(10,10))
    divider = make_axes_locatable(ax)
    df_merged.plot(column='class', cmap='tab20', linewidth=1, ax=ax, legend = True)
    ax.axis('off')
    st.pyplot(fig)    

    st.markdown(
        """
        ### Rank Clusters
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

        ahp.node('a-1', 'City Cluster 1')
        ahp.node('a-2', 'City Cluster 2')
        ahp.node('a-3', 'City Cluster 3')
        ahp.node('a-4', 'City Cluster 4')


        ahp.edge('goal', 'c-demo-GDP')
        ahp.edge('goal', 'c-demo-pd')

        ahp.edge('goal', 'c-env-summer')
        ahp.edge('goal', 'c-env-winter')
        ahp.edge('goal', 'c-env-climate')

        ahp.edge('goal', 'c-eco-income')
        ahp.edge('goal', 'c-eco-charging')
                
        ahp.edge('goal', 'c-trans-beh')

        for i in [str(i) for i in range(1,5)]:
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
        st.write("""
        The number of cluster here is for illustrative purpose.
        The preference matrix of criteria is
        """)

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
                weighted_sum = np.dot(pref_matrix, weights)
                consistency_vector = weighted_sum / weights
                lambda_max = np.mean(consistency_vector)
                CI = (lambda_max-n)/(n-1)
                RI = {1:0, 2:0, 3:0.58, 4:0.9, 5:1.12, 6:1.24, 7:1.32, 8:1.41, 9:1.45, 10:1.49, 11:1.51, 12:1.48, 13:1.56, 14:1.57, 15:1.58}
                if n > 15:
                    RI[n] = round(1.98 * (n - 2) / n, 2)
                CR = CI/RI[n]
                if CR>0.1:
                    st.warning("Preference matrix used in AHP is inconsistent! CR is "+str(CR))
                weights.rename("Weight", inplace=True)
                return weights
        
            weights=ahp_get_weights(df_a)
        
        st.write("Now let's evaluate the alternatives")
        import numpy as np

        def generate_pairwise_matrix_from_data(data, preference='higher'):
            """
            Generates a pairwise comparison matrix from quantitative data.

            Parameters:
            - data: dataframe, index are cluster IDs, values are median GDP per Capita.
            - preference: 'higher' if higher values are preferred, 'lower' if lower values are preferred.

            Returns:
            - pairwise_matrix: numpy array
            - cluster_ids: list of cluster IDs
            """
            cluster_ids = list(data.index)
            n = len(cluster_ids)
            pairwise_matrix = np.ones((n, n))
            data = data.replace(0, 1e-6)
            if preference == 'lower':
                data = data.max() + data.min() - data

            # Normalize data to [1, 9]
            min_value = data.min()
            max_value = data.max()
            normalized_data = 1 + 8 * (data - min_value) / (max_value - min_value)

            for i in range(n):
                for j in range(i + 1, n):
                    ratio = normalized_data.iloc[i] / normalized_data.iloc[j]
                    pairwise_matrix[i, j] = ratio
                    pairwise_matrix[j, i] = 1 / ratio

            return pairwise_matrix, cluster_ids


        dict_pref_alt_keys = agg_df.columns[3:-1]
        dict_pref_alt = {}
        for key in dict_pref_alt_keys:
            df = pd.concat([agg_df.groupby('class')[key].median(),agg_df_outliers_1st_pca[[key]]])
            if key in ["Env-P-2", "Env-C-1", "T-P-1", "D-EPTE-1"]:
                matrix, matrix_index = generate_pairwise_matrix_from_data(df, preference="lower")
            else:
                matrix, matrix_index = generate_pairwise_matrix_from_data(df, preference="higher")
            dict_pref_alt[key] = pd.DataFrame(matrix, columns = matrix_index, index = matrix_index)

        crit = st.selectbox("Check the preference matrix and weights of alternatives for a criterion", df_a.index)
        column1, column2 = st.columns([2,1])
        with column1:
            df_crit = pd.DataFrame(dict_pref_alt[crit])
            df_crit
        with column2:
            df_weights_crit_alt = ahp_get_weights(dict_pref_alt[crit])
            st.write(df_weights_crit_alt)

        st.write(
            "Final scores for each alternative "
        )

        dfs = []

        for crit in df_a.index:
            df = ahp_get_weights(dict_pref_alt[crit])
            df.rename("Weights " + crit, inplace=True)
            dfs.append(df)
        df_all = pd.concat(dfs, axis=1)
        final_scores = (df_all*weights.values.T).sum(axis=1)
        final_scores.sort_values(ascending=False, inplace=True)
        final_scores.rename("Final Scores", inplace=True)
        final_scores
        
        agg_df_outliers_1st_pca['class']=agg_df_outliers_1st_pca.index
        agg_df=pd.concat([agg_df, agg_df_outliers_1st_pca], axis=0)
        agg_df = agg_df.merge(final_scores, left_on='class', right_index=True)

        df_merged = gdf.merge(agg_df, left_index=True, right_index=True)


    fig, ax = plt.subplots(1, figsize=(10,10))
    divider = make_axes_locatable(ax)
    df_merged.plot(column='Final Scores', cmap='Oranges', linewidth=1, ax=ax, legend=False)
    ax.axis('off')
    st.pyplot(fig)

    st.markdown(
        """
        ### Estimate adoption rate for each cluster
        """
    )

    final_score_scaled = final_scores/final_scores.max()

    max_adoption_rate = st.number_input("The maximum adoption rate is ", value=0.6, min_value=0.0, max_value=1.0)
    adoption_rates = max_adoption_rate*final_score_scaled
    adoption_rates.rename("Adoption Rates", inplace=True)
    st.write(
        """
        Adoption rate for each cluster is then
        """, adoption_rates)
    

    # csv = convert_df(adoption_rates)
    # st.download_button("Download here", csv, mime='text/csv', file_name='adoption_rates_clusters_geo.csv')

    # geo_info = df_merged.merge(adoption_rates, how='left', left_on='class', right_index=True)[["class", "adop_rate"]]
    # geo_info

with tab2:
    csv_file = st.file_uploader("Upload all vehicles included in Serviceable Available Market")
    if csv_file is not None:
        df_vehicle = pd.read_csv(csv_file)
    else:
        file_path = fr"{precharged_folder}\{str(current_year)}_vehicle_list_SAM.csv"
        df_vehicle = pd.read_csv(file_path,low_memory=False, index_col=0, dtype=object)
    df_vehicle = change_type(df_vehicle)
    st.write(df_vehicle.head())
    st.write(
        """
        ### Cluster
        """
    )
    with st.expander("Click here to see how clustering is done on vehicle models"):
        """
        #### Group by
        The vehicle dataframe is grouped by type_version_variante and calculates specific aggregations for each column in the group:
        - `poids_a_vide_national`, `puissance_net_maxi`, `co2`, and `Age`: Calculates the median value.
        - `carrosserie_ce`: Finds the most frequent (mode) value.
        The `classe_env` is not kept as it is highly correlated with the column `Age`.
        """

        df_tvv = df_vehicle[["type_version_variante", "puissance_net_maxi", "poids_a_vide_national", "carrosserie_ce", "co2", "Age"]].groupby(by='type_version_variante').agg(
            {"poids_a_vide_national": 'median',
            "puissance_net_maxi": 'median',
            "carrosserie_ce": lambda x:x.mode().iloc[0],
            "co2": 'median',
            "Age": "median"}
        )
        st.write(df_tvv)
        """
        #### Encoding
        This encoding step assigns numeric scores to categorical variables to make them suitable for analysis. 
        For `carrosserie_ce`, we use label encoding to assign unique integer codes to each category, making the data ready for further quantitative analysis.
        Here is its encoding dictionary. 
        """
        st.write(dict(enumerate(df_tvv['carrosserie_ce'].astype('category').cat.categories)))
        df_tvv['carrosserie_ce'] = df_tvv['carrosserie_ce'].astype('category').cat.codes
        st.write(df_tvv)
        
        scaler = StandardScaler()
        df_tvv_scaled = pd.DataFrame(scaler.fit_transform(df_tvv), columns=df_tvv.columns)

        dict_score = find_num_clusters(df_tvv_scaled)
        st.write("The silhouette scores for each number of clusters are", dict_score)
        n=int(st.number_input("The number of clusters chosen here is ", value=10))
        kmeans = KMeans(n_clusters = n, init='k-means++', random_state=0).fit(df_tvv_scaled)
        df_tvv_scaled["class"] = kmeans.labels_
        pca=PCA()
        df_tvv_2d = pd.DataFrame(pca.fit_transform(df_tvv_scaled.iloc[:,:-1]))
        df_tvv_2d.columns = ['PC'+str(i) for i in range(1,6)]
        df_tvv_2d.index = df_tvv_scaled.index
        df_tvv_2d = df_tvv_2d[['PC1', 'PC2']]
        df_tvv_2d['class'] = df_tvv_scaled['class']
        st.plotly_chart(px.scatter(df_tvv_2d, x='PC1', y='PC2', color='class'))

    """
    ### Rank Clusters
    """
    with st.expander("Click here to see how ranking of clusters is done on vehicle models"):
        st.write("""
        The criteria involved here are: `poids_a_vide_national`, `puissance_net_maxi`, `carrosserie_ce`, `co2`, `Age`.
        
        The preference matrix proposed is:
        """)
        criteria = df_tvv.columns
        df_pref_vehicle = pd.DataFrame([
            [1, 5, 3, 7, 3],
            [1/5, 1, 1/3, 3, 1/3],
            [1/3, 3, 1, 5, 1],
            [1/7, 1/3, 1/5, 1, 1/5],
            [1/3, 3, 1, 5, 1]
        ], index=criteria, columns=criteria)
        st.write(df_pref_vehicle)

        weights_crit_vehicle = ahp_get_weights(df_pref_vehicle)
        st.write("The weights for each criteria are: ",weights_crit_vehicle)
        df_tvv["class"] = kmeans.labels_
        st.write("The statistics for the different vehicle model clusters are shown as below. The median values for numerical attributes are calculated, and the most common category for the `carrosserie_ce` is shown.", df_tvv.groupby(by='class').agg(
            {"poids_a_vide_national":"median",
            "puissance_net_maxi":"median",
            "carrosserie_ce":lambda x:x.mode().iloc[0],
            "co2":"median",
            "Age":"median"}
        ))

        






with tab3:
    csv_file = st.file_uploader("Upload all vehicles included in Serviceable Available Market here")
    if csv_file is not None:
        df_vehicle = pd.read_csv(csv_file)
    else:
        file_path = fr"{precharged_folder}\{str(current_year)}_vehicle_list_SAM.csv"
        df_vehicle = pd.read_csv(file_path,low_memory=False, index_col=0, dtype=object)
    
    df_vehicle = df_vehicle.merge(geo_info, how='left', left_on="code_commune_titulaire", right_index=True)
    df_vehicle['retrofit_service'] = np.nan
    for i in range(df_vehicle.shape[0]):
        proba_retrofit = df_vehicle.iloc[i]["adop_rate"]
        df_vehicle.iloc[i,-1] = np.random.choice([False,True], p=[1-proba_retrofit, proba_retrofit])
    
    st.write(df_vehicle)

if st.button("Go to DEC"):
    st.switch_page("MarketSharefortheCompany.py")
