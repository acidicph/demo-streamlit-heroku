#Core Pkg
import streamlit as st
import os 
import joblib


#EDA Pkgs
import pandas as pd 
import numpy as np 

#Data Vis Pkgs
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

predict_label = {'Patient NOT likely to be readmitted within 30 days.':0, 'Patient requires attention, likely to be admitted within 30 days!!':1}


def preprocessing(df):

    features = ['time_in_hospital',
     'num_procedures',
     'num_medications',
     'number_outpatient',
     'number_emergency',
     'number_inpatient',
     'num_lab_procedures',
     'number_diagnoses','metformin',
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
     'payer_self', 'age_cat_0-30', 'age_cat_30-50','age_cat_50-70','age_cat_>70']


    numerics = ['time_in_hospital',
     'num_procedures',
     'num_medications',
     'number_outpatient',
     'number_emergency',
     'number_inpatient',
     'num_lab_procedures',
     'number_diagnoses']


    df2=df.copy()
    
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(df2[numerics])

    import pickle
    scalerfile = 'scaler.sav'
    pickle.dump(scaler, open(scalerfile, 'wb'))
    scaler = pickle.load(open(scalerfile, 'rb'))
    df2[numerics] = scaler.transform(df2[numerics])

    inputdata =pd.concat([ df['encounter_ID'], df[features]], axis=1)
    scaleddata = df2[features]
   
    return inputdata, scaleddata
 


def main():

	st.title("Admit Once")
	st.subheader("Reducing Readmission Rate for Patients with Diabetes")
	image = Image.open('icon.png')
	st.image(image, caption='Developed by Ebrahim Ghazvini', use_column_width=True)
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

		if st.checkbox("Value Count Plot"):
			st.write(data['time_in_hospital'].value_counts().plot(kind='bar'))
			st.pyplot()

	if choices == "Predict":
		st.subheader("Predict")

		uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
		if uploaded_file is not None:
		   	data = pd.read_csv(uploaded_file)
		   	st.write(data)
	

		model_choice = st.selectbox("Model Choice", ['LogisticRegression'])
		if st.button("Evaluate"):
    		
    			if model_choice == 'LogisticRegression':
    				predictor = load_prediction_model("LogReg_model.sav")
    				inputdata, scaleddata = preprocessing(data)

    				prediction = predictor.predict(scaleddata)
    				prediction = pd.DataFrame(prediction)
    				prediction = prediction.rename({0:'Prediction Results'}, axis=1)
    				encidpred = pd.concat([ inputdata['encounter_ID'], prediction], axis=1)
    				st.subheader("""Prediction key:
    					"0" means patient not likely to be readmitted within 30 days of discharge.
    					"1" means patient likely to be readmitted within 30 days.
    					""")
    				st.write(encidpred)
    				#final_result = get_key(prediction['Prediction Results'], predict_label)
    				#st.success(final_result)
    				
    				res =pd.DataFrame(predictor.predict_proba(scaleddata) )
    				res['Readm Prob']=res[1]
    				res2 = pd.concat([ res['Readm Prob'], inputdata], axis=1)
    				#res2 = res2.sort_values(1, ascending=False)
    				result=res2
    				st.subheader("""Probability of a patient to be readmitted:
    					
    					""")
    				st.write(result)

	if choices == "Simulate":
		st.subheader("Simulate")

if __name__ == '__main__':
	main()




