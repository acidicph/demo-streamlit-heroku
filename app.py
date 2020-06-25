
import streamlit as st
import os
import joblib
import st_state_patch

# EDA Pkgs
import pandas as pd
import numpy as np

# Data Vis Pkgs
import matplotlib
import altair as alt

matplotlib.use('Agg')

from PIL import Image

# connect to sql lite
import sqlite3
from sqlite3 import Connection

URI_SQLITE_DB = "admit_once.db"

features = ['time_in_hospital',
                        'num_procedures',
                        'num_medications',
                        'number_outpatient',
                        'number_emergency',
                        'number_inpatient',
                        'num_lab_procedures',
                        'number_diagnoses', 'metformin',
                        'repaglinide',
                        'nateglinide',
                        'chlorpropamide',
                        'glimepiride',
                        'glipizide',
                        'glyburide',
                        'pioglitazone',
                        'rosiglitazone',
                        'insulin',
                        'glyburide-metformin',
                        'change',
                        'diabetesMed',
                        'gender_Male',
                        'admission_type_id_urgent',
                        'admission_source_id_er',
                        'admission_source_id_referral',
                        'admission_source_id_transfer',
                        'max_glu_serum_>200',
                        'max_glu_serum_>300',
                        'max_glu_serum_Norm',
                        'A1Cresult_>7',
                        'A1Cresult_>8',
                        'A1Cresult_Norm',
                        'diag_t_circulatory',
                        'diag_t_digestive',
                        'diag_t_metabolic_immunity',
                        'diag_t_respiratory',
                        'diag_t2_circulatory',
                        'diag_t2_digestive',
                        'diag_t2_genitourinary',
                        'diag_t2_metabolic_immunity',
                        'diag_t2_respiratory',
                        'admit_phys_cadio',
                        'admit_phys_er',
                        'admit_phys_gen_surgery',
                        'admit_phys_internal_med',
                        'discharge_to_another_rehab',
                        'discharge_to_home',
                        'discharge_to_home_health_serv',
                        'discharge_to_inpatient_inst',
                        'discharge_to_others',
                        'discharge_to_short_hospital',
                        'discharge_to_snf',
                        'payer_blu_x',
                        'payer_health_maint_org',
                        'payer_medicaid',
                        'payer_medicare',
                        'payer_self', 'age_cat_0-30', 'age_cat_30-50', 'age_cat_50-70', 'age_cat_>70']

@st.cache
def load_prediction_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file), 'rb'))
    return loaded_model


def get_key(val, my_dict):
    for key, value in my_dict.items():
        if val == value:
            return key


def preprocessing(df_new, ID2):
    df = pd.read_csv("sampletest.csv")
    df.loc[ID2] =df_new
    df = pd.get_dummies(df, columns=['weight', 'race', 'gender', 'admission_type_id',
                                     'admission_source_id', 'max_glu_serum', 'A1Cresult', 'readmitted', 'diag_t',
                                     'diag_t2', 'admit_phys', 'discharge_to', 'payer', 'age_cat'])

    numerics = ['time_in_hospital',
                'num_procedures',
                'num_medications',
                'number_outpatient',
                'number_emergency',
                'number_inpatient',
                'num_lab_procedures',
                'number_diagnoses']

    #df = df.loc[df['encounter_id'] == ID2]
    
    df2 = df.copy()
    df2 = df2.loc[ID2]
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(df2[numerics])

    import pickle
    scalerfile = 'scaler.sav'
    scaler = pickle.load(open(scalerfile, 'rb'))
    df2[numerics] = scaler.fit(df2[numerics]).transform(df2[numerics])

    df =df.loc[ID2]

    inputdata = pd.concat([df['encounter_id'], df[features]], axis=1)

    scaleddata = df2[features]

    return inputdata, scaleddata



def preprocessing_main(df_new, ID2):
    df = pd.read_csv("sampletest.csv")
    df = pd.get_dummies(df, columns=['weight', 'race', 'gender', 'admission_type_id',
                                     'admission_source_id', 'max_glu_serum', 'A1Cresult', 'readmitted', 'diag_t',
                                     'diag_t2', 'admit_phys', 'discharge_to', 'payer', 'age_cat'])

    numerics = ['time_in_hospital',
                'num_procedures',
                'num_medications',
                'number_outpatient',
                'number_emergency',
                'number_inpatient',
                'num_lab_procedures',
                'number_diagnoses']

    df = df.loc[df['encounter_id'] == ID2]
    
    df2 = df.copy()
    #df2 = df2.loc[ID2]
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(df2[numerics])

    import pickle
    scalerfile = 'scaler.sav'
    scaler = pickle.load(open(scalerfile, 'rb'))
    df2[numerics] = scaler.fit(df2[numerics]).transform(df2[numerics])

    #df =df.loc[ID2]

    inputdata = pd.concat([df['encounter_id'], df[features]], axis=1)

    scaleddata = df2[features]

    return inputdata, scaleddata



@st.cache(hash_funcs={Connection: id})
def get_connection(path: str):
    """Put the connection in cache to reuse if path does not change between Streamlit reruns.  """
    return sqlite3.connect(path, check_same_thread=False)


def main():
    conn = get_connection(URI_SQLITE_DB)
    tabs_style = """
<style>
        * {
  box-sizing: border-box;
}
body {
  font-family: "Saira";
  background: #DCE3E8;
  color: #697c86;
  line-height: 1.618em;
}
.wrapper {
  max-width: 50rem;
  width: 100%;
  margin: 0 auto;
}
.tabs {
  position: relative;
  margin: 3rem 0;
  background: #ff787f;
  height: 14.75rem;
}
.tabs::before,
.tabs::after {
  content: "";
  display: table;
}
.tabs::after {
  clear: both;
}
.tab {
  float: left;
}
.tab-switch {
  display: none;
}
.tab-label {
  position: relative;
  display: block;
  line-height: 2.75em;
  height: 3em;
  padding: 0 1.618em;
  background: #ff787f;
  border-right: 0.125rem solid #16a085;
  color: #fff;
  cursor: pointer;
  top: 0;
  transition: all 0.25s;
}
.tab-label:hover {
  top: -0.25rem;
  transition: top 0.25s;
}
.tab-content {
  height: 12rem;
  position: absolute;
  z-index: 1;
  top: 2.75em;
  left: 0;
  padding: 1.618rem;
  background: #fff;
  color: #2c3e50;
  border-bottom: 0.25rem solid #bdc3c7;
  opacity: 0;
  transition: all 0.35s;
}
.tab-switch:checked + .tab-label {
  background: #fff;
  color: #ff787f;
  border-bottom: 0;
  border-right: 0.125rem solid #fff;
  transition: all 0.35s;
  z-index: 1;
  top: -0.0625rem;
}
.tab-switch:checked + label + .tab-content {
  z-index: 2;
  opacity: 1;
  transition: all 0.35s;
}
.ReactVirtualized__Grid 
{
    background-color: #FFB6C1 !important;
}
.dataframe  { color:black;}
.sidebar-content { background-color: #754155;} 
</style>
    """
    df = pd.read_csv("sampletest_sim.csv")
    df['index'] = df['encounter_id']
    df = df.set_index('index')

    df_limited = pd.read_csv("sampletest.csv")
    df_limited['index'] = df_limited['encounter_id']
    df_limited = df_limited.set_index('index')
    st.write(tabs_style, unsafe_allow_html=True)


    image = Image.open('icon2.png')
    st.image(image, use_column_width=True)

    menu = ["Explore", "Simulate"]
    choices = st.sidebar.selectbox("Select from Menu", menu)

    if choices == "Explore":
        st.subheader("Explore")

        st.dataframe(df.head(10))

        if st.checkbox("Show Summary"):
            st.write(df.describe())

        if st.checkbox("Probabiliy Density"):
            predictor = load_prediction_model("LogReg_model.sav")
            inputdata, scaleddata = preprocessing_main(df_limited, df_limited.index)
            prediction = predictor.predict(scaleddata)
            res = pd.DataFrame(predictor.predict_proba(scaleddata))
            res['Readm Prob'] = res[1]
            st.write(res['Readm Prob'].plot(kind='hist', x="Current Probabiliy of Patient Readmission", y="Fraction",
                                            title='Current Probabiliy Distribution of Patient Readmission', bins=50))
            st.pyplot()

    if choices == "Simulate":
        st.subheader("""Simulate:
            1. Select Patient ID
            2. Adjust Variables 
            3. Click Simulate button.
            """)
        df_limited = pd.read_csv("sampletest_sim.csv")
        df_limited['index'] = df_limited['encounter_id']
        df_limited = df_limited.set_index('index')
        
        selected_indicies = st.multiselect('Select from the list of inpatients below:', df_limited.index)
        selected_rows =  df_limited.loc[selected_indicies]
        #selected_rows = selected_rows.drop(columns=['readmitted'])
        st.write('### _Selected inpatient encounter ID_', selected_rows)
        s = st.State()
        if not s:
            s.pressed_first_button = False

    if st.button("Populate data of selected inpatient") or s.pressed_first_button:
        s.pressed_first_button = True  # preserve the info that you hit a button between runs
        st.markdown("# _Length_ _of_ _Stay_:")
        new_time_in_hospital = st.slider("Time in hospital", 1, 14, int(selected_rows['time_in_hospital']),
                                         None, None)

        st.markdown("# Medications")
        new_num_medications = st.slider("Number of Medication", 1, 81, int(selected_rows['num_medications']))

        meds = ["Metformin", "Repaglinide", "Nateglinide", "Chlorpropamide", "Glimepiride", "Glipizide",
                "Glyburide", "Pioglitazone", "Insulin", "Glyburide-Mtformin"]
        med_choice = st.selectbox("Select medication from Menu", meds)
        options = [0, 1, 2, 3]
        med_options = st.selectbox("Dose", options,
                                   format_func=lambda x: "Not on this medication" if x == 0 else (
                                       "Lower dosage" if x == 1 else (
                                           "Steady dosage" if x == 2 else "Increase dosage")))

        if med_choice == "Metformin":
            selected_rows.loc[:, 'metformin'] = med_options

        if med_choice == "Repaglinide":
            selected_rows.loc[:, 'repaglinide'] = med_options

        if med_choice == "Nateglinide":
            selected_rows.loc[:, 'nateglinide'] = med_options

        if med_choice == "Chlorpropamide":
            selected_rows.loc[:, 'chlorpropamide'] = med_options

        if med_choice == "Glimepiride":
            selected_rows.loc[:, 'glimepiride'] = med_options

        if med_choice == "Glipizide":
            selected_rows.loc[:, 'glipizide'] = med_options

        if med_choice == "Glyburide":
            selected_rows.loc[:, 'glyburide'] = med_options

        if med_choice == "Pioglitazone":
            selected_rows.loc[:, 'pioglitazone'] = med_options

        if med_choice == "Insulin":
            selected_rows.loc[:, 'insulin'] = med_options

        if med_choice == "Glyburide-Metformin":
            selected_rows.loc[:, 'glyburide-metformin'] = med_options

        st.markdown("# Discharge:")
        st.markdown(" _Select_ _where_ _to_ _discharge_ _inpatient_:")

        discharge = ["Home","Skilled Nursing Facility", "Home Health Service", "Rehab Center", 
                     "Transfer to another hospital", "Short Stay Unit"]
        # discharge_option = ['snf']
        discharge_to = st.selectbox("Select from Menu", discharge)
        if discharge_to == "Home":
            selected_rows.loc[:, 'discharge_to'] = ['discharge_to_home']
        if discharge_to == "Skilled Nursing Facility":
            selected_rows.loc[:, 'discharge_to'] = ['snf']
        if discharge_to == "Home Health Service":
            selected_rows.loc[:, 'discharge_to'] = ['discharge_to_home_health_serv']
        if discharge_to == "Rehab Center":
            selected_rows.loc[:, 'discharge_to'] = ['discharge_to_another_rehab']
        
        if discharge_to == "Transfer to another hospital":
            selected_rows.loc[:, 'discharge_to'] = ['discharge_to_inpatient_inst']
        if discharge_to == "Short Stay Unit":
            selected_rows.loc[:, 'discharge_to'] = ['discharge_to_short_hospital']

        selected_rows.loc[:, 'time_in_hospital'] = new_time_in_hospital
        selected_rows.loc[:, 'num_medications'] = new_num_medications

        st.markdown("## Summary of New input")
        st.write(selected_rows)

        if st.button("Simulate"):
            df3 = selected_rows
            predictor = load_prediction_model("LogReg_model.sav")
            inputdata, scaleddata = preprocessing(df3, selected_indicies)
            prediction = predictor.predict(scaleddata)
            prediction = pd.DataFrame(prediction)
            prediction = prediction.rename({0: 'Prediction Results'}, axis=1)
            prediction['Prediction Results'] = prediction['Prediction Results']
            encidpred = pd.concat([inputdata['encounter_id'].reset_index(drop=True), prediction], axis=1)
            encidpred = encidpred.replace({1: "Yes", 0: "No"})
            st.markdown("## Simulated Prediction")

            res = pd.DataFrame(predictor.predict_proba(scaleddata))
            res['Readmission Probability'] = res[1]
            res['Risk'] = res['Readmission Probability'].apply(lambda pred:"High Risk" if pred > 0.75 else ("Medium Risk" if pred >= 0.5 else "Low Risk"))
            st.write(pd.concat([encidpred, pd.DataFrame(res['Readmission Probability']).reset_index(drop=True), pd.DataFrame(res['Risk']).reset_index(drop=True)], axis=1))
            #res2 = pd.concat([res['Readmission Probability'], res['Risk'], inputdata.reset_index(drop=True)], axis=1)
            # res2 = res2.sort_values(1, ascending=False)
            #result = res2
            #st.markdown(" ## Simulated Probability of a patient to be readmitted:")
            #st.write(result)

            coeflist = predictor.coef_
            coeflist = np.transpose(coeflist)
            columns = ["coef"]
            multiplier = pd.DataFrame(index=features, data=coeflist, columns=columns)
            multiplier.index.name = 'Variables'
            multiplier.reset_index(inplace=True)
            multiplier = multiplier.sort_values(by='coef', ascending=False)
            valuecoef = np.transpose(scaleddata)
            valuecoef.columns = ['values']
            valuecoef.index.name = 'Variables'
            valuecoef.reset_index(inplace=True)
            mergetab = pd.merge(multiplier, valuecoef, on='Variables')
            mergetab['Impact'] = mergetab['values'] * mergetab['coef']
            mergetab['absimpact'] = abs(mergetab['Impact'])
            mergetab = mergetab[mergetab['Impact'] > 0]
            mergetab = mergetab.sort_values(by='absimpact', ascending=False)
            mergetab.index = mergetab['Variables']
            #mergetab = mergetab.drop(['age_cat_30-50', 'age_cat_50-70', 'insulin', 'num_medications', 'num_procedures',                 'repaglinide'])
            mergetab = mergetab[mergetab['absimpact'] >= .005]
            st.write(alt.Chart(mergetab).mark_bar(size=20).encode(
                x=alt.X('Impact'), y=alt.X('Variables', sort=None), ).configure_axis(labelFontSize=20,
                                                                                     titleFontSize=20))




if __name__ == '__main__':
    main()

