# Core Pkg
import csv
from _csv import writer

import streamlit as st
import os
import joblib
import st_state_patch

# EDA Pkgs
import pandas as pd
import numpy as np

# Data Vis Pkgs
import matplotlib.pyplot as plt
import matplotlib
import altair as alt

matplotlib.use('Agg')
import seaborn as sns
from PIL import Image


@st.cache
def load_data(dataset):
    df = pd.read_csv(dataset)
    return df


def load_prediction_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file), 'rb'))
    return loaded_model


def get_key(val, my_dict):
    for key, value in my_dict.items():
        if val == value:
            return key


predict_label = {'Patient NOT likely to be readmitted within 30 days.': 0,
                 'Patient requires attention, likely to be admitted within 30 days!!': 1}


def preprocessing(df, ID2):

    df=pd.get_dummies(df, columns=['weight', 'race', 'gender', 'admission_type_id',
                                 'admission_source_id', 'max_glu_serum', 'A1Cresult', 'readmitted', 'diag_t',
                                 'diag_t2', 'admit_phys', 'discharge_to', 'payer', 'age_cat'])

    df=df.drop(columns=['age','discharge_disposition_id', 'payer_code', 'medical_specialty', 'weight_unkown',
                        'race_?','race_Other','gender_Unknown/Invalid','admission_type_id_Not Mapped',
                        'admission_type_id_unknown', 'admission_source_id_missing', 'admission_source_id_other',
                        'max_glu_serum_None','A1Cresult_None',  'admit_phys_others', 'admit_phys_unkown',
                        'payer_others','readmitted_>30', 'readmitted_NO'])

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

    numerics = ['time_in_hospital',
                'num_procedures',
                'num_medications',
                'number_outpatient',
                'number_emergency',
                'number_inpatient',
                'num_lab_procedures',
                'number_diagnoses']

    df= df.loc[df['encounter_ID']==ID2]
    df2 = df.copy()
    

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(df2[numerics])

    import pickle
    scalerfile = 'scaler.sav'
    pickle.dump(scaler, open(scalerfile, 'wb'))
    scaler = pickle.load(open(scalerfile, 'rb'))
    df2[numerics] = scaler.transform(df2[numerics])

    inputdata = pd.concat([df['encounter_ID'], df[features]], axis=1)
    scaleddata = df2[features]

    return inputdata, scaleddata


def main():
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
</style>
    """
    st.write(tabs_style, unsafe_allow_html=True)

    #st.title("Admit Once")  

    image = Image.open('icon2.png')
    st.image(image, use_column_width=True)
    # Menu
    menu = ["Explore", "Predict", "Simulate"]
    choices = st.sidebar.selectbox("Select from Menu", menu)

    if choices == "Explore":
        st.subheader("Explore")

        data = load_data("diabetes2.csv")
        st.dataframe(data.head(10))

        if st.checkbox("Show Summary"):
            st.write(data.describe())

        if st.checkbox("Show Shape"):
            st.write(data.shape)

        #if st.checkbox("Distribution of Patients' Time in Hospital"):
            #st.write(data['time_in_hospital'].value_counts().plot(kind='bar', x= "Number of Days", y= "Number of Patients",title = 'Distribution of Patients\' time in hospital'))
            #st.pyplot()
            #data['time_in_hospital']\
            #.plot.bar(rot=0)\
            #.set(title="Distribution of Patients' Time in Hospital",
            #xlabel="Days", ylabel="No. of Patients)")
            #st.pyplot()


        data2 = load_data("cleaned3.csv")
        if st.checkbox("Probabiliy Density"):
            predictor = load_prediction_model("LogReg_model2.sav")
            prediction = predictor.predict(data2)
            res = pd.DataFrame(predictor.predict_proba(data2))
            res['Readm Prob'] = res[1]
            #res2 = pd.concat([res['Readm Prob'], inputdata.reset_index(drop=True)], axis=1)
            st.write(res['Readm Prob'].plot(kind='hist', x= "Current Probabiliy of Patient Readmission", y= "Fraction",title = 'Current Probabiliy Distribution of Patient Readmission', bins=50))
            st.pyplot()


    if choices == "Predict":
        st.subheader("Predict")

        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.write(data)

        if st.button("Evaluate"):

                predictor = load_prediction_model("LogReg_model2.sav")
                inputdata, scaleddata = preprocessing(data)

                prediction = predictor.predict(scaleddata)
                prediction = pd.DataFrame(prediction)
                prediction = prediction.rename({0: 'Prediction Results'}, axis=1)
                encidpred = pd.concat([inputdata['encounter_ID'], prediction], axis=1)
                st.subheader("""Prediction key:
    					"0" means patient not likely to be readmitted within 30 days of discharge.
    					"1" means patient likely to be readmitted within 30 days.
    					""")
                st.write(encidpred)
                # final_result = get_key(prediction['Prediction Results'], predict_label)
                # st.success(final_result)

                res = pd.DataFrame(predictor.predict_proba(scaleddata))
                res['Readm Prob'] = res[1]
                res2 = pd.concat([res['Readm Prob'], inputdata], axis=1)
                # res2 = res2.sort_values(1, ascending=False)
                result = res2
                st.subheader("""Probability of a patient to be readmitted:""")
                st.write(result)

    if choices == "Simulate":
        st.subheader("""Simulate:
            1. Enter Patient ID
            2. Adjust Variables 
            3. Click Simulate button.
            """)
        file_name = "sampletest.csv"
        dfsim = pd.read_csv("sampletest.csv")
        input_id = st.text_input("Enter ID", "")
        s = st.State()
        if not s:
            s.pressed_first_button = False

        if st.button("Search by Encounter ID")  or s.pressed_first_button:
            s.pressed_first_button = True  # preserve the info that you hit a button between runs
            try:
                IDS = [input_id]
                ID2= int(input_id)
                df3 = dfsim.loc[dfsim['encounter_ID']==ID2]
                df3 = df3.drop(columns = ['readmitted'])
                st.markdown("### Current Status")
                st.dataframe(df3.style.highlight_max(axis=1))
                

                if input_id !="":

                    with open(file_name, newline='') as csvfile:
                        reader = csv.DictReader(csvfile)
                        rows = [row for row in reader if row['encounter_ID'] in IDS]
                        
                        df3 = dfsim.loc[dfsim['encounter_ID']==ID2]
                        
                        
                    if rows:

                        st.markdown("# _Length_ _of_ _Stay_:")
                        new_time_in_hospital = st.slider("Time in hospital", 1, 14, int(rows[0]['time_in_hospital']), None, None)
                        
                        st.markdown("# Medications")
                        new_num_medications = st.slider("Number of Medication", 1, 81, int(rows[0]['num_medications']))

                        meds = ["Metformin", "Repaglinide", "Nateglinide", "Chlorpropamide", "Glimepiride", "Glipizide", "Glyburide", "Pioglitazone", "Insulin", "Glyburide-Mtformin"]
                        med_choice = st.selectbox("Select medication from Menu", meds)
                        options =  [0, 1, 2, 3]
                        med_options = st.selectbox("Dose", options, format_func=lambda x: "Not on this medication" if x == 0 else ("Lower dosage" if x == 1 else ("Steady dosage" if x == 2 else "Increase dosage")) )
                        
                        if med_choice == "Metformin":
                            df3.loc[:, 'metformin'] =  med_options
                        

                        if med_choice == "Repaglinide":
                            df3.loc[:, 'repaglinide'] =  med_options

                        if med_choice == "Nateglinide":
                            df3.loc[:, 'nateglinide'] =  med_options
                        
                        if med_choice == "Chlorpropamide":
                            df3.loc[:, 'chlorpropamide'] =  med_options
                        
                        if med_choice == "Glimepiride":
                            df3.loc[:, 'glimepiride'] =  med_options
                        
                        if med_choice == "Glipizide":
                            df3.loc[:, 'glipizide'] =  med_options
                        
                        if med_choice == "Glyburide":
                            df3.loc[:, 'glyburide'] =  med_options
                        
                        if med_choice == "Pioglitazone":
                            df3.loc[:, 'pioglitazone'] =  med_options
                        
                        if med_choice == "Insulin":
                            df3.loc[:, 'insulin'] =  med_options
                        
                        if med_choice == "Glyburide-Metformin":
                            df3.loc[:, 'glyburide-metformin'] =  med_options


                        st.markdown("# Discharge:")
                        st.markdown(" _Select_ _where_ _to_ _discharge_ _inpatient_:")

                        discharge = ["Skilled Nursing Facility", "Home Health Service", "Rehab Center", "Home", "Transfer to another hospital", "Short Stay Unit"]
                        #discharge_option = ['snf']
                        discharge_to = st.selectbox("Select from Menu", discharge)

                        if discharge_to == "Skilled Nursing Facility":
                            df3.loc[:, 'discharge_to'] =  ['snf']
                        if discharge_to == "Home Health Service":
                            df3.loc[:, 'discharge_to'] =  ['discharge_to_home_health_serv']
                        if discharge_to == "Rehab Center":
                            df3.loc[:, 'discharge_to'] =  ['discharge_to_another_rehab']
                        if discharge_to == "Home":
                            df3.loc[:, 'discharge_to'] =  ['discharge_to_home']
                        if discharge_to == "Transfer to another hospital":
                            df3.loc[:, 'discharge_to'] =  ['discharge_to_inpatient_inst']
                        if discharge_to == "Short Stay Unit":
                            df3.loc[:, 'discharge_to'] =  ['discharge_to_short_hospital']


                        df3.loc[:, 'time_in_hospital'] =  new_time_in_hospital
                        df3.loc[:, 'num_medications'] =  new_num_medications
                        
                        st.markdown("## Summary of New input")
                        st.write(df3)


                    if st.button("Simulate"):
                        dfsim.loc[dfsim['encounter_ID']==ID2] = df3
                        predictor = load_prediction_model("LogReg_model2.sav")
                        inputdata, scaleddata = preprocessing(dfsim, ID2)
                        prediction = predictor.predict(scaleddata)
                        prediction = pd.DataFrame(prediction)
                        prediction = prediction.rename({0: 'Prediction Results'}, axis=1)
                        encidpred = pd.concat([inputdata['encounter_ID'].reset_index(drop=True), prediction], axis=1)
                        encidpred = encidpred.replace({1: "Yes", 0: "No"})
                        st.markdown("## Simulated Prediction")
                       
                        res = pd.DataFrame(predictor.predict_proba(scaleddata))
                        res['Readm Prob'] = res[1]
                        st.write(pd.concat([encidpred,pd.DataFrame(res['Readm Prob']).reset_index(drop=True)], axis=1) )
                        res2 = pd.concat([res['Readm Prob'], inputdata.reset_index(drop=True)], axis=1)
                        # res2 = res2.sort_values(1, ascending=False)
                        result = res2
                        st.markdown(" ## Simulated Probability of a patient to be readmitted:")
                        #st.write(result)
                        
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

                        coeflist=predictor.coef_
                        coeflist=np.transpose(coeflist)
                        columns=["coef"]
                        multiplier=pd.DataFrame(index=features ,data=coeflist , columns=columns)
                        multiplier.index.name = 'Variables'
                        multiplier.reset_index(inplace=True)
                        multiplier = multiplier.sort_values(by='coef', ascending=False)
                        valuecoef = np.transpose(scaleddata)
                        valuecoef.columns = ['values']
                        valuecoef.index.name = 'Variables'
                        valuecoef.reset_index(inplace=True)
                        mergetab = pd.merge(multiplier,valuecoef, on='Variables')
                        mergetab['Impact'] = mergetab['values']*mergetab['coef']
                        mergetab['absimpact'] = abs(mergetab['Impact'])
                        mergetab = mergetab.sort_values(by='absimpact', ascending=False)
                        mergetab.index=mergetab['Variables']
                        mergetab = mergetab.drop(['age_cat_30-50','age_cat_50-70','insulin','num_medications','num_procedures','repaglinide'])
                        mergetab = mergetab[mergetab['absimpact'] >= .005]
                        st.write(alt.Chart( mergetab).mark_bar(size=20).encode(
                        x=alt.X('Impact'), y=alt.X('Variables', sort=None),).configure_axis(labelFontSize=20,titleFontSize=20).properties(width=700, height=500) )
                
            except Exception as e:
                e = 'Looks like I am having problem connecting my backend'
                st.write(e)
        else:
            st.write("None found!")


if __name__ == '__main__':
    main()
