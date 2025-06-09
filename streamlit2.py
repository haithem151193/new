import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sns


data=pd.read_csv('Financial_inclusion_dataset.csv') # import data
data1=data.copy() # copy of data
data.drop(columns=['uniqueid'], inplace=True)  # drop uniqueid
data1.drop(columns=['uniqueid'], inplace=True)  # drop uniqueid

for column in data1.columns:
    if data1[column].dtype == "int64" or data1[column].dtype == "float64":
        data1[column].fillna(data1[column].median(), inplace=True)
    else:
        data1[column].fillna(data1[column].mode()[0], inplace=True)
# Data Cleaning

for column in data.columns:
    if data[column].dtype == "int64" or data[column].dtype == "float64":
        data[column].fillna(data[column].median(), inplace=True)
    else:
        data[column].fillna(data[column].mode()[0], inplace=True)

# Data Transformation ( transform data and mapping dictionary)

label_encoder = preprocessing.LabelEncoder()

data['gender_of_respondent']= label_encoder.fit_transform(data['gender_of_respondent'])
mapping_dict_gender_of_respondent = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
data['cellphone_access']= label_encoder.fit_transform(data['cellphone_access'])
mapping_dict_cellphone_access = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
data['relationship_with_head'] = label_encoder.fit_transform(data['relationship_with_head'])
mapping_dict_relationship_with_head = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
data['marital_status'] = label_encoder.fit_transform(data['marital_status'])
mapping_dict_marital_status = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
data['job_type'] = label_encoder.fit_transform(data['job_type'])
mapping_dict_job_type = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
data['education_level'] = label_encoder.fit_transform(data['education_level'])
mapping_dict_education_level = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
data['location_type'] = label_encoder.fit_transform(data['location_type'])
mapping_dict_location_type = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
data['country'] = label_encoder.fit_transform(data['country'])
mapping_dict_country = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
data['bank_account'] = label_encoder.fit_transform(data['bank_account'])
mapping_bank_account = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))


x = data.drop(['bank_account'], axis=1, errors='ignore') #input
y = data['bank_account'] # output
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1000) # split data into train and test
model = LogisticRegression()  # import Logistic regression
model.fit(x_train, y_train) # train model
st.title("Welcome to my app") # display title

choice = st.selectbox('Select your choice', ('Visualize', 'predict')) # create selectbox containing visualize / predict

if (choice == 'Visualize'):
        status = st.selectbox('Select graph you want to visulize', ('','pie_chart_bank_account', 'Bank Account Ownership by Year',
        'Bank Account Ownership by location_type' ,'Bank Account Ownership by cellphone_access' , 'Bank Account Ownership by gender_of_respondent',
        'Bank Account Ownership by relationship_with_head' , 'Bank Account Ownership by education_level' , 'Bank Account Ownership by marital_status',
        'Bank Account Ownership by job_type'))
        if (status == ''):
                st.error("Error")
                st.write('choose a plot')
        elif (status == 'pie_chart_bank_account'):
                fig, ax = plt.subplots()
                plt.pie(data1['bank_account'].value_counts(), labels=['Yes', 'No'], autopct="%1.1f%%")
                plt.title('Bank Account Ownership')
                st.pyplot(fig)
        elif (status == 'Bank Account Ownership by Year'):
                fig, ax = plt.subplots()
                sns.countplot(x='year', hue='bank_account', data=data1, ax=ax)
                st.pyplot(fig)
        elif (status == 'Bank Account Ownership by location_type'):
                fig, ax = plt.subplots()
                sns.countplot(x='location_type', hue='bank_account', data=data1, ax=ax)
                st.pyplot(fig)
        elif (status == 'Bank Account Ownership by cellphone_access'):
                fig, ax = plt.subplots()
                sns.countplot(x='cellphone_access', hue='bank_account', data=data1, ax=ax)
                st.pyplot(fig)
        elif (status == 'Bank Account Ownership by gender_of_respondent'):
                fig, ax = plt.subplots()
                sns.countplot(x='gender_of_respondent', hue='bank_account', data=data1, ax=ax)
                st.pyplot(fig)
        elif (status == 'Bank Account Ownership by relationship_with_head'):
                fig, ax = plt.subplots()
                sns.countplot(x='relationship_with_head', hue='bank_account', data=data1, ax=ax)
                st.pyplot(fig)
        elif (status == 'Bank Account Ownership by education_level'):
                fig, ax = plt.subplots()
                sns.countplot(x='education_level', hue='bank_account', data=data1, ax=ax)
                st.pyplot(fig)
        elif (status == 'Bank Account Ownership by marital_status'):
                fig, ax = plt.subplots()
                sns.countplot(x='marital_status', hue='bank_account', data=data1, ax=ax)
                st.pyplot(fig)
        elif (status == 'Bank Account Ownership by job_type'):
                fig, ax = plt.subplots()
                sns.countplot(x='job_type', hue='bank_account', data=data1, ax=ax)
                st.pyplot(fig)
else:

        country = st.selectbox('Select your country', ("Rwanda","Tanzania","Kenya","Uganda"))
        country_map = mapping_dict_country[country]
        year = st.selectbox('Select your year', (2016,2017,2018))
        location_type = st.selectbox('Select your location_type', set(data1['location_type']))
        location_type_map = mapping_dict_location_type[location_type]
        cellphone_access = st.selectbox('Select your cellphone_access', ("Yes","No"))
        cellphone_access_map = mapping_dict_cellphone_access[cellphone_access]
        household_size=st.number_input('enter a household_size' , step=1)
        age_of_respondent=st.number_input('enter an age_of_respondent' , step=1)
        gender_of_respondent = st.selectbox('Select the Gender of respondent', ("Male","Female"))
        gender_of_respondent_map = mapping_dict_gender_of_respondent[gender_of_respondent]
        relationship_with_head = st.selectbox('Select your relationship_with_head', ("Head of Household","Spouse",
                                                                        "Child","Parent","Other relative"
                                                                        "Other non-relatives"))
        relationship_with_head_map = mapping_dict_relationship_with_head[relationship_with_head]
        marital_status = st.selectbox('Select your marital_status', ("Married/Living together","Single/Never Married",
                                                                        "Widowed","Divorced/Seperated","Dont know"))
        marital_status_map = mapping_dict_marital_status[marital_status]
        education_level = st.selectbox('Select your education_level', ("Primary education","No formal education",
                                                                "Secondary education","Tertiary education",
                                                               "Vocational/Specialised training","Other/Dont know/RTA"))
        education_level_map = mapping_dict_education_level[education_level]
        job_type = st.selectbox('Select your job_type', ("Self employed","Informally employed","Farming and Fishing",
                                                 "Remittance Dependent","Other Income" ,"Formally employed Private",
                                                  "No Income","Formally employed Government", "Government Dependent",
                                                 "Dont Know/Refuse to answer"))
        job_type_map = mapping_dict_job_type[job_type]
        my_dict0 = my_dict ={'country': country, 'year' : year , 'location_type' : location_type,
         'cellphone_access': cellphone_access , 'household_size' : household_size,
          'age_of_respondent': age_of_respondent , 'gender_of_respondent':gender_of_respondent,
          'relationship_with_head': relationship_with_head , 'marital_status':marital_status ,
          'education_level': education_level , 'job_type': job_type}
        my_dict ={'country': country_map, 'year' : year , 'location_type' : location_type_map,
         'cellphone_access': cellphone_access_map , 'household_size' : household_size,
          'age_of_respondent': age_of_respondent , 'gender_of_respondent':gender_of_respondent_map,
          'relationship_with_head': relationship_with_head_map , 'marital_status':marital_status_map,
          'education_level': education_level_map , 'job_type': job_type_map}
        data_test= pd.DataFrame(my_dict, index = [0])
        data_test0 = pd.DataFrame(my_dict0, index = [0])

        if (st.button('Show my entry')):
                st.write(data_test0) # write entries
        elif (st.button('predict')):
                y_pred_test=(model.predict_proba(data_test)[:, 1] >= 0.7).astype(int)
                if y_pred_test == 0:
                     st.write('Your possible bank account state  is No')
                else:
                     st.write('Your possible bank account state  is Yes')