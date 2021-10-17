import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf
import sklearn
import keras
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
from sklearn.metrics import accuracy_score



def app():   
    

    st.title("Medical dataset prediction using ML")
    st.write("""
    #Explore different ML algorithms
    """)
    datasets = st.sidebar.selectbox("Select Datasets",("Heart Disease","Diabetes","BP"))
    st.write(datasets)


    if datasets =="Heart Disease" or datasets=="Diabetes" or datasets=="BP":
        classifiers = st.sidebar.selectbox("Select Classifier",("KNN","SVC","Decision Tree","Random Forest"))
        st.write(classifiers)
        
        def load_datasets_csv(datasets):
            if datasets == "Heart Disease":
                data = pd.read_csv("heart.csv")
                st.image("2.png")
                st.write("The term “heart disease” refers to several types of heart conditions. The most common type of heart disease in the United States is coronary artery disease (CAD), which affects the blood flow to the heart. Decreased blood flow can cause a heart attack.")
                st.write("Your Age ")
                age = st.number_input("age",0,100)
                st.write("Your Gender Female:0 , Male : 1")
                Gender = st.number_input("Gender",0,1)
                st.write("cp 0 to 3")
                cp = st.number_input("cp",0,3)
                st.write("trestbps")
                trestbps = st.number_input("trestbps",100,300)
                st.write("chol")
                chol = st.number_input("chol",100,400)
                st.write("fbs  0 or 1")
                fbs = st.number_input("fbs",0,1)
                st.write("restecg 0 or 1")
                restecg = st.number_input("restecg",0,1)
                st.write("thalach")
                thalach = st.number_input("thalach",100,400)
                st.write("exang 0 or 1")
                exang = st.number_input("exang",0,1)
                st.write("oldpeak(decimal)")
                oldpeak = st.number_input("oldpeak",0.0,9.0)
                st.write("slope 0 to 2")
                slope = st.number_input("slope",0,2)
                st.write("ca 0 or 1")
                ca = st.number_input("ca",0,1)
                st.write("thal 1 t0 3")
                thal = st.number_input("thal",1,3)
                data1={'Age':age, 'Gender':Gender, 'cp':cp, 'trestbps':trestbps,
                    'chol':chol, 'fbs':fbs, 'restecg':restecg, 'thalach':thalach,
                    'exang':exang, 'oldpeak':oldpeak, 'slope':slope, 'ca':ca,
                    'thal':thal}

            elif datasets == "Diabetes":
                data = pd.read_csv("Diabetes.csv")
                st.image('4.png')
                st.write("Diabetes is a disease that occurs when your blood glucose, also called blood sugar, is too high. Blood glucose is your main source of energy and comes from the food you eat. Insulin, a hormone made by the pancreas, helps glucose from food get into your cells to be used for energy.")
                no_times_pregnant = st.number_input("no_times_pregnant",0,20)
                glucose_concentration = st.number_input("glucose_concentration",0,1000)
                blood_pressure = st.number_input("blood_pressure",0,1000)
                skin_fold_thickness = st.number_input("skin_fold_thickness",0,1000)
                serum_insulin = st.number_input("serum_insulin",0,1000)
                bmi = st.number_input("bmi",0.0,1000.0)
                diabetespedigree = st.number_input("diabetespedigree",0.0,30.0)
                age = st.number_input("age",0,200)
                data1={'no_times_pregnant':no_times_pregnant, 'glucose_concentration ':glucose_concentration , 'blood_pressure':blood_pressure, 'skin_fold_thickness':skin_fold_thickness,
                    'serum_insulin':serum_insulin, 'bmi':bmi, 'diabetespedigree':diabetespedigree, 'age':age}
            elif datasets == "BP":
                data = pd.read_csv("BP.csv")
                st.image('5.png')
                st.write("Blood pressure is the pressure of circulating blood against the walls of blood vessels. Most of this pressure results from the heart pumping blood through the circulatory system. When used without qualification, the term blood pressure refers to the pressure in the large arteries. ")
                st.write("Level_of_Hemoglobin")
                Level_of_Hemoglobin = st.number_input("Level_of_Hemoglobin",0.0,40.0)
                st.write("Genetic_Pedigree_Coefficient")
                Genetic_Pedigree_Coefficient = st.number_input("Genetic_Pedigree_Coefficient",0.0,40.0)
                st.write("Age")
                Age = st.number_input("Age",0,100)
                st.write("BMI")
                BMI = st.number_input("BMI",0,100)
                st.write("Sex 0 or 1")
                Sex = st.number_input("Sex",0,1)
                st.write("Smoking 0 or 1")
                Smoking = st.number_input("Smoking",0,1)
                st.write("Physical_activity")
                Physical_activity = st.number_input("Physical_activity",0,1000000)
                st.write("salt_content_in_the_diet")
                salt_content_in_the_diet = st.number_input("salt_content_in_the_diet",0,1000000)        
                st.write("alcohol_consumption_per_day")
                alcohol_consumption_per_day = st.number_input("alcohol_consumption_per_day",0,1000000)
                st.write("Level_of_Stress 1 to 3")
                Level_of_Stress = st.number_input("Level_of_Stress",1,3)
                st.write("Chronic_kidney_disease 0 or 1")
                Chronic_kidney_disease = st.number_input("Chronic_kidney_disease",0,1)
                st.write("Adrenal_and_thyroid_disorders 0 or 1")
                Adrenal_and_thyroid_disorders = st.number_input("Adrenal_and_thyroid_disorders",0,1)
            
                data1={'Level_of_Hemoglobin':Level_of_Hemoglobin, 'Genetic_Pedigree_Coefficient ':Genetic_Pedigree_Coefficient , 'Age':Age, 'BMI':BMI,
                    'Sex':Sex, 'Smoking':Smoking, 'Physical_activity':Physical_activity, 'salt_content_in_the_diet':salt_content_in_the_diet,
                    'alcohol_consumption_per_day':alcohol_consumption_per_day, 'Level_of_Stress':Level_of_Stress, 'Chronic_kidney_disease':Chronic_kidney_disease, 
                    'Adrenal_and_thyroid_disorders':Adrenal_and_thyroid_disorders}
                
            return data , data1
        data , data1 = load_datasets_csv(datasets)
        data = data.dropna() # If any null values are present in dataset it will be drop
        n_f = data.select_dtypes(include=[np.number]).columns
        c_f = data.select_dtypes(include=[np.object]).columns
        data = pd.get_dummies(data)
        X = data.drop(data.iloc[:,-1:],axis=1)
        y = data.iloc[:,-1:]
        
        def add_parameters_csv(clf_name):
            p = dict()
            if clf_name == "KNN":
                K = st.sidebar.slider("K",1,30)
                p["K"] = K
            elif clf_name == "SVC":
                C = st.sidebar.slider("C",0.01,15.0)
                p["C"] = C
            elif clf_name == "Random Forest":
                max_depth = st.sidebar.slider("max_depth",2,15)
                n_estimators = st.sidebar.slider("n_estimators",1,100)
                p["max_depth"] = max_depth
                p["n_estimators"] = n_estimators
            else:
                min_samples_split = st.sidebar.slider("min_samples_split",2,5)
                p["min_samples_split"] = min_samples_split
            return p
        p = add_parameters_csv(classifiers)
        
        def get_Classifier_csv(clf_name,p):
            if clf_name == "KNN":
                clf = KNeighborsClassifier(n_neighbors=p["K"])
            elif clf_name == "SVC":
                clf = SVC(C=p["C"])
            elif clf_name == "Random Forest":
                clf = RandomForestClassifier(n_estimators=p["n_estimators"],max_depth=p["max_depth"],random_state=1200)
            else:
                clf = DecisionTreeClassifier(min_samples_split=p["min_samples_split"])
            return clf
        clf = get_Classifier_csv(classifiers,p)


        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1200)
        clf.fit(X_train,y_train)
        y_pred_test = clf.predict(X_test)
        acc = accuracy_score(y_test,y_pred_test)
        st.write(f"classifier Used={classifiers}")
        st.write(f"accuracy score={acc}")
        df2 = pd.DataFrame(data1,index=["Name"])
        y_pred_test1 = clf.predict(df2)
        st.write(y_pred_test1)
        if y_pred_test1 ==  0:    
            st.write("No")
            st.write(datasets)
        else:     
            st.write("Yes")
            st.write(datasets)
        
        
        