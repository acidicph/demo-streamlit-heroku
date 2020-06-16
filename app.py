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


def preprocessing(df):
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


        data2 = load_data("cleaned2.csv")
        if st.checkbox("Probabiliy Density"):
            predictor = load_prediction_model("LogReg_model.sav")
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

                predictor = load_prediction_model("LogReg_model.sav")
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
        file_name = "train_input_test.csv"
        dfsim = pd.read_csv("train_input_test.csv")
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



                if input_id !="":

                    with open(file_name, newline='') as csvfile:
                        reader = csv.DictReader(csvfile)
                        rows = [row for row in reader if row['encounter_ID'] in IDS]
                        st.markdown("### Current Status")
                        df3 = dfsim.loc[dfsim['encounter_ID']==ID2]
                        st.write(df3)
                        
                    if rows:

                        st.markdown("# Length of Stay:")
                        new_time_in_hospital = st.slider("Time in hospital", 1, 14, int(rows[0]['time_in_hospital']), None, None)
                        st.markdown("# Medications")
                        new_num_medications = st.slider("Number of Medication", 1, 81, int(rows[0]['num_medications']))
                        st.markdown("### Key: 0: N/A, 1: Lower Dose, 2: Steady Does, 3: Increase Dose")
                        new_metform = st.slider("Metformin", 0,3, int(rows[0]['metformin']))
                        new_repaglinide = st.slider("Repaglinide", 0,3, int(rows[0]['repaglinide']))
                        new_nateglinide = st.slider("Nateglinide", 0,3, int(rows[0]['nateglinide']))
                        new_chlorpropamide = st.slider("Chlorpropamide", 0,3, int(rows[0]['chlorpropamide']))
                        new_glimepiride= st.slider("Glimepiride", 0,3, int(rows[0]['glimepiride']))
                        new_glipizide = st.slider("Glipizide", 0,3, int(rows[0]['glipizide']))
                        new_glyburide = st.slider("Glyburide", 0,3, int(rows[0]['glyburide']))
                        new_pioglitazone = st.slider("Pioglitazone", 0,3, int(rows[0]['pioglitazone']))
                        new_insulin = st.slider("Insulin", 0,3, int(rows[0]['insulin']))
                        new_glyburide_metformin = st.slider("Glyburide-Mtformin", 0,3, int(rows[0]['glyburide-metformin']))
                        st.markdown("# Discharge:")
                        #st.markdown("### Key: 0:No, 1: Yes")
                        #Rehab = st.slider("Rehab Center", 0,1, int(rows[0]['discharge_to_another_rehab']))
                        #Home = st.slider("Home", 0,1, int(rows[0]['discharge_to_home']))
                        #Health_service = st.slider("Home Health Service", 0,1, int(rows[0]['discharge_to_home_health_serv']))
                        #Hospital = st.slider("Another Hospital", 0,1, int(rows[0]['discharge_to_inpatient_inst']))
                        #Short = st.slider("Short Stay Unit", 0,1, int(rows[0]['discharge_to_short_hospital']))
                        #snf = st.slider("Skilled Nursing Facilit", 0,1, int(rows[0]['discharge_to_snf']))
                        snf = st.selectbox("Skilled Nursing Facilit",(0,1), int(rows[0]['discharge_to_snf']), format_func=lambda x: "Yes" if x == 1 else "No")
                        Health_service = st.selectbox("Home Health Service", (0,1), int(rows[0]['discharge_to_home_health_serv']), format_func=lambda x: "Yes" if x == 1 else "No")
                        Rehab = st.selectbox("Rehab Center", (0,1), int(rows[0]['discharge_to_home']), format_func=lambda x: "Yes" if x == 1 else "No")
                        Home = st.selectbox("Home", (0,1), int(rows[0]['discharge_to_snf']), format_func=lambda x: "Yes" if x == 1 else "No")
                        Hospital = st.selectbox("Another Hospital", (0,1), int(rows[0]['discharge_to_inpatient_inst']), format_func=lambda x: "Yes" if x == 1 else "No")
                        Short = st.selectbox("Short Stay Unit", (0,1), int(rows[0]['discharge_to_short_hospital']), format_func=lambda x: "Yes" if x == 1 else "No")



                        df3 = dfsim.loc[dfsim['encounter_ID']==ID2]


                        df3.loc[:, 'time_in_hospital'] =  new_time_in_hospital
                        df3.loc[:, 'num_medications'] =  new_num_medications
                        df3.loc[:, 'metformin'] =  new_metform
                        df3.loc[:, 'repaglinide'] =  new_repaglinide
                        df3.loc[:, 'nateglinide'] =  new_nateglinide
                        df3.loc[:, 'chlorpropamide'] =  new_chlorpropamide
                        df3.loc[:, 'glimepiride'] =  new_glimepiride
                        df3.loc[:, 'glipizide'] =  new_glipizide
                        df3.loc[:, 'glyburide'] =  new_glyburide
                        df3.loc[:, 'pioglitazone'] =  new_pioglitazone
                        df3.loc[:, 'insulin'] =  new_insulin
                        df3.loc[:, 'glyburide-metformin'] =  new_glyburide_metformin
                        df3.loc[:, 'discharge_to_another_rehab'] =  Rehab
                        df3.loc[:, 'discharge_to_home'] =  Home
                        df3.loc[:, 'discharge_to_home_health_serv'] =  Health_service
                        df3.loc[:, 'discharge_to_short_hospital'] =  Short
                        df3.loc[:, 'discharge_to_snf'] = snf
                        df3.loc[:, 'discharge_to_inpatient_inst'] =  Hospital
                        st.markdown("## Summary of New input")
                        st.write(df3)
                        
                        


                        if st.button("Simulate"):   
                            predictor = load_prediction_model("LogReg_model.sav")
                            inputdata, scaleddata = preprocessing(df3)

                            prediction = predictor.predict(scaleddata)
                            prediction = pd.DataFrame(prediction)
                            prediction = prediction.rename({0: 'Prediction Results'}, axis=1)
                            encidpred = pd.concat([inputdata['encounter_ID'].reset_index(drop=True), prediction], axis=1)
                            encidpred = encidpred.replace({1: "Yes", 0: "No"})
                            st.markdown("## Simulated Prediction")
                            st.write(encidpred)
                # final_result = get_key(prediction['Prediction Results'], predict_label)
                # st.success(final_result)

                            res = pd.DataFrame(predictor.predict_proba(scaleddata))
                            res['Readm Prob'] = res[1]
                            res2 = pd.concat([res['Readm Prob'], inputdata.reset_index(drop=True)], axis=1)
                            # res2 = res2.sort_values(1, ascending=False)
                            result = res2
                            st.markdown(" ## Simulated Probability of a patient to be readmitted:")
                            st.write(result)
                            

                        


                    else:
                            st.write("None found!")

                else:
                    st.write("Please set an Id first.")
            except Exception as e:
                # 'Looks like I am having problem connecting my backend' +
                st.write(e)






if __name__ == '__main__':
    main()
