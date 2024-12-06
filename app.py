# import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from pathlib import Path
#shiny
from shiny.express import ui, input
from shiny.ui import page_navbar
from shiny import render, reactive



#read in the file
df = pd.read_csv(Path(__file__).parent / "Data" / "diabetes_data_upload.csv")
# encode
df.replace({'Yes': 1, 'No': 0, 'Positive': 1, 'Negative': 0}, inplace=True)

# drop class bacause that is our target
X = df.drop(columns=['class'])
y = df['class']


#create subset data
df_male=df[df['Gender']=="Male"]
df_female=df[df['Gender']=="Female"]



#create webapp
with ui.navset_pill(id="tab"):
    with ui.nav_panel("Introduction & Data Sources"):  
        #data source
        with ui.layout_columns():  
            with ui.card():  
                ui.card_header("Introduction")
                ui.p("Diabetes is a significant global health issue that can lead to severe complications if not detected early. Early identification of individuals at risk for diabetes can aid in prompt interventions, lifestyle adjustments, and potentially prevent the onset of diabetes. Our project aims to create a web app that uses a model to predict the probability of an individual being at risk for diabetes based on multiple attributes. Using easily measurable, self-reported data, we aim to build a model that will educate users about diabetes risk factors.")

            with ui.card():  
                ui.card_header("Dataset Source")
                ui.p("We are using a publicly available dataset from Kaggle that includes variables such as age, sex, polyuria, polydipsia, obesity, etc. Link to dataset: https://www.kaggle.com/datasets/ishandutta/early-stage-diabetes-risk-prediction-dataset")


    with ui.nav_panel("Demographic Statistics"): 
        ui.input_select(  
        "Gender",  
        "Select a Gender below:",  
        {"All":"All", "Male": "Male", "Female": "Female"},  
        ) 

        @render.plot  
        def plot():  
            if input.Gender()=="Male":
                output_data=df_male
            elif input.Gender()=="Female":
                output_data=df_female
            else:
                output_data=df
        
            ax = sns.boxplot(x='class', y='Age', data=output_data) 
            ax.set_title("Age Distribution by Diabetes Class")
            ax.set_xlabel("Diabetes Class (0 = Negative, 1 = Positive)")
            ax.set_ylabel("Age")
            return ax  

    with ui.nav_panel("Most Common Symptoms"):  
        #create data frame for symptoms
        symptoms_data={'Symptom':['Polyuria', 'Polydipsia', 'Weakness', 'Partial Paresis','Sudden Weight Loss', 'Polyphagia', 'Visual Blurring','Itching', 'Delayed Healing', 'Muscle Stiffness','Genital Thrush', 'Irritability', 'Alopecia', 'Obesity'], 'Percentage of Diabetes Cases with Symptom':[0.76, 0.70, 0.68, 0.60, 0.59, 0.59, 0.55, 0.48, 0.48, 0.42, 0.34, 0.26, 0.24, 0.19]}
        symptoms_df=pd.DataFrame(symptoms_data)

        @render.text  
        def text():
            return "This ranking shows that Polyuria and Polydipsia are the most prominent symptoms among diabetics, each present in over 70% of cases."

        @render.data_frame  
        def output_df():
            return render.DataTable(symptoms_df) 

    with ui.nav_panel("Predictive Model"):  
        #inputs
        ui.input_numeric("Age", "Enter Your Age", 40, min=1, max=150)  

        ui.input_select(  
            "Gender_1",  
            "Select a Gender option below:",  
            {"M": "Male", "F": "Female"},  
        )    

        @render.text  
        def text1():
            return "Select the symptoms that you have"

        ui.input_checkbox("Polyuria", "Polyuria", False) 
        ui.input_checkbox("Polydipsia", "Polydipsia", False)  
        ui.input_checkbox("swl", "Sudden Weight Loss", False)  
        ui.input_checkbox("weakness", "Weakness", False)  
        ui.input_checkbox("Polyphagia", "Polyphagia", False)  
        ui.input_checkbox("gt", "Genital Thrush", False)  
        ui.input_checkbox("vb", "Visual Blurring", False)  
        ui.input_checkbox("Itching", "Itching", False)  
        ui.input_checkbox("Irritability", "Irritability", False)  
        ui.input_checkbox("dh", "Delayed Healing", False)  
        ui.input_checkbox("pp", "Partial Paresis", False)  
        ui.input_checkbox("ms", "Muscle Stiffness", False)  
        ui.input_checkbox("Alopecia", "Alopecia", False)  
        ui.input_checkbox("Obesity", "Obesity", False) 
        ui.input_action_button("btn", "Predict")

        @reactive.calc
        def lr():
            # encode categorical variables
            X_encoded = pd.get_dummies(X, drop_first=True)
            # create and train the logistic regression model
            log_reg = LogisticRegression(max_iter=1000, random_state=42)
            log_reg.fit(X_encoded, y)

            if input.Gender_1()=='M':
                log_odd=1/(1+np.exp(-(log_reg.intercept_[0]+log_reg.coef_[0,0]*input.Age()+log_reg.coef_[0,1]*input.Polyuria()+log_reg.coef_[0,2]*input.Polydipsia()+log_reg.coef_[0,3]*input.swl()+log_reg.coef_[0,4]*input.weakness()+log_reg.coef_[0,5]*input.Polyphagia()+log_reg.coef_[0,6]*input.gt()+log_reg.coef_[0,7]*input.vb()+log_reg.coef_[0,8]*input.Itching()+log_reg.coef_[0,9]*input.Irritability()+log_reg.coef_[0,10]*input.dh()+log_reg.coef_[0,11]*input.pp()+log_reg.coef_[0,12]*input.ms()+log_reg.coef_[0,13]*input.Alopecia()+log_reg.coef_[0,14]*input.Obesity()+log_reg.coef_[0,15])))
                round_log_odd=round(log_odd*100,2)
                output=str(round_log_odd)+"%"
            else:
                log_odd=1/(1+np.exp(-(log_reg.intercept_[0]+log_reg.coef_[0,0]*input.Age()+log_reg.coef_[0,1]*input.Polyuria()+log_reg.coef_[0,2]*input.Polydipsia()+log_reg.coef_[0,3]*input.swl()+log_reg.coef_[0,4]*input.weakness()+log_reg.coef_[0,5]*input.Polyphagia()+log_reg.coef_[0,6]*input.gt()+log_reg.coef_[0,7]*input.vb()+log_reg.coef_[0,8]*input.Itching()+log_reg.coef_[0,9]*input.Irritability()+log_reg.coef_[0,10]*input.dh()+log_reg.coef_[0,11]*input.pp()+log_reg.coef_[0,12]*input.ms()+log_reg.coef_[0,13]*input.Alopecia()+log_reg.coef_[0,14]*input.Obesity())))
                round_log_odd=round(log_odd*100,2)
                output=str(round_log_odd)+"%"
            return output
        
        @render.code
        @reactive.event(input.btn)
        def txt2():
            return f"The chance that you have Diabetes is: {lr()}"

    with ui.nav_panel("Input Variable Importance"):  
        @render.text  
        def text5():
            return "Below is the ranked importance of input variables in our model:"
        
        import_data={'Rank': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16], 'Feature':['Polydipsia','Polyuria','Gender','Irritability','Itching','Genital Thrush','Partial Paresis','Polyphagia','Sudden Weight Loss','Visual Blurring','Weakness','Delayed Healing','Muscle Stiffness','Obesity','Alopecia','Age']}
        import_df=pd.DataFrame(import_data)

        @render.data_frame  
        def output_df1():
            return render.DataTable(import_df) 
