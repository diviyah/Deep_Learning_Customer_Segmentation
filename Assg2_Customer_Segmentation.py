# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 09:41:41 2022

@author: diviyah
"""

import os
import pickle
import datetime
import numpy as np
import pandas as pd
import missingno as msno
import scipy.stats as ss
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import plot_model
from modules_Assg2_CS import EDA,ModelCreation,ModelEvaluation,CramersV


#%% STATICS

DF_FILE_PATH=os.path.join(os.getcwd(),'dataset','train.csv')
log_dir=datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
LOG_PATH=os.path.join(os.getcwd(),'logs',log_dir)
JOB_TYPE_PICKLE_PATH=os.path.join(os.getcwd(),'model','job_type.pkl')
MARITAL_PICKLE_PATH=os.path.join(os.getcwd(),'model','marital.pkl')
EDUCATION_PICKLE_PATH=os.path.join(os.getcwd(),'model','education.pkl')
DEFAULT_PICKLE_PATH=os.path.join(os.getcwd(),'model','default.pkl')
HOUSING_LOAN_PICKLE_PATH=os.path.join(os.getcwd(),'model','housing_loan.pkl')
PERSONAL_LOAN_PICKLE_PATH=os.path.join(os.getcwd(),'model','personal_loan.pkl')
COMMUNICATION_TYPE_PICKLE_PATH=os.path.join(os.getcwd(),'model','communication_type.pkl')
MONTH_PICKLE_PATH=os.path.join(os.getcwd(),'model','month.pkl')
PREV_CAMPAIGN_OUTCOME_PICKLE_PATH=os.path.join(os.getcwd(),'model','prev_campaign_outcome.pkl')
CS_MODEL_SAVE_PATH = os.path.join(os.getcwd(),'model','csnn_model.h5')

#%% Data Loading

df=pd.read_csv(DF_FILE_PATH)

#%% Data Visualizations to

column_names=['id','customer_age','job_type','marital','education','default',
                'balance','housing_loan','personal_loan','communication_type',
                'day_of_month','month','last_contact_duration',
                'num_contacts_in_campaign','days_since_prev_campaign_contact',
                'num_contacts_prev_campaign','prev_campaign_outcome',
                'term_deposit_subscribed']

# Data Identified into categorical and continuous data
cont_data =['customer_age','balance','day_of_month','last_contact_duration',
            'num_contacts_in_campaign','days_since_prev_campaign_contact',
            'num_contacts_prev_campaign']
                
cat_data=['job_type','marital','education','default','housing_loan','personal_loan',
          'communication_type','month','prev_campaign_outcome','term_deposit_subscribed']



eda=EDA()

eda.cont_plot(cont_data,df)

# Observations:
#   - Customer age is mostly focused between the age of 25--40
#   - Bank balance indicates most people possesses fewer bank balance value
#   - day of month for banking is reaching high and low
#   - The last contact duration is reaching its peak from .....
#   -

eda.cat_plot(cat_data,df)

# Observations:
#   job_type: respondents were mostly blue-collar, management and the least from unknown category.
#   marital: Most were married
#   default: most answered no
#   housing_loan: most took the housing loan 
#   personal_loan: lots of them possess personal loan
#   communication_type: most uses cellular while telephone is at the least
#   month: Month of May reached the highest and the lowest being December
#   pre_campaign_outcome: most response obtained were unknown with success being very low
#   term_deposit_subscribed: most picked 0


eda.cat_hue_plot(cat_data,df)

eda.groupby_plot(cat_data,df)


## OBSERVATIONS:
# job_type: Mode is blur-collar,management chose 0 and followed by technician and fewer chose 1
#  marital: Mode is 0, especially by marital followed by single
#  default: Mode is 0, by no for default
#  h_loan:  Mode is 0, Majority of has housing loan.
#  p_loan:  Mode is 0, of for 0 of personal loan
#  comm:    Mode is 0, for 0 of comm type
#  month:   Mode is 0 for 8th month(August)
#  prev_camp: Mode is 0 for 3rd responded for prev camp outcome
     
#%% Data Inspections

df.info() # There are lots of missing values
          # We have object, int64 and float64 datatype

temp=df.describe().T    #The difference between mean and median quite huge 
                        # Indication of presence of outliers and non-normal distribution of the data


# Check for abnormalities categorical datas
for i in cat_data:
    print(i,':',df[i].unique())


# Check for NaNs
msno.bar(df) #lots of missing values for 'days_since_prev_campaign_contact'
             # can consider this variable for better accuracy
df.isna().sum()
        # NaNs in customer_age,marital,balance,personal_loan,last_contact_duration,
        #.......... num_contacts_in_campaign, days_since_prev_campaign_contact
        # There's HUGE NaNs in days_since_prev_campaign_contact

# Check for duplicates
df.duplicated().sum() #0 duplicates


#%% Label Encoder


# To Numerize categorical columns
le=LabelEncoder()

pickle_path=[JOB_TYPE_PICKLE_PATH,
             MARITAL_PICKLE_PATH,
             EDUCATION_PICKLE_PATH,
             DEFAULT_PICKLE_PATH,
             HOUSING_LOAN_PICKLE_PATH,
             PERSONAL_LOAN_PICKLE_PATH,
             COMMUNICATION_TYPE_PICKLE_PATH,
             MONTH_PICKLE_PATH,
             PREV_CAMPAIGN_OUTCOME_PICKLE_PATH]


# Covert categorical data into integers with NaN in the df on display
for index,i in enumerate(cat_data):
    temp=df[i]
    temp[temp.notnull()]=le.fit_transform(temp[temp.notnull()])
    df[i]=pd.to_numeric(temp,errors='coerce')
    
# Saving the pickle file
for index,i in enumerate(pickle_path):
        with open(pickle_path[index],'wb') as file:
            pickle.dump(le,file)


#%% Data Cleaning


# Drop 2 cols of data
#   - dropped data id because its irrelevant information to what we analyse
#   - dropped 'days_since_prev_campaign_contact' because there are lots of NaN value
#   - Even if we impute the variable, it not gonna give a good accuracy

df=df.drop(labels=['id','days_since_prev_campaign_contact'],axis=1)

# updated column_names and cont_data after removed the 2 variables
column_names=['customer_age','job_type','marital','education','default',
                'balance','housing_loan','personal_loan','communication_type',
                'day_of_month','month','last_contact_duration',
                'num_contacts_in_campaign','num_contacts_prev_campaign',
                'prev_campaign_outcome','term_deposit_subscribed']
                
cont_data =['customer_age','balance','day_of_month','last_contact_duration',
            'num_contacts_in_campaign','num_contacts_prev_campaign']

# Using MICE Imputation method

ii = IterativeImputer()
df = ii.fit_transform(df)
df=pd.DataFrame(df)
# the name of the columns are gone, so add them.
df.columns=['customer_age','job_type','marital','education','default',
                'balance','housing_loan','personal_loan','communication_type',
                'day_of_month','month','last_contact_duration',
                'num_contacts_in_campaign','num_contacts_prev_campaign',
                'prev_campaign_outcome','term_deposit_subscribed']

df.isna().sum() # No Nans anymore

df.info()
        # We have data imputed BUT its in float
        # Hence, need to convert them into int (compulsory for categorical data)
        # Data Identified into categorical and continuous data


for index, i in enumerate(column_names):
    df[i]= np.floor(df[i]).astype('int')

# no more decimal values in our data. 
# We can leave our balance variable to be float due to its relevance (nature of money value). 

#%% Feature Selection

# continuous vs categorical data

lr=LogisticRegression()

for i in cont_data:
    print(i)
    lr.fit(np.expand_dims(df[i],axis=-1),df['term_deposit_subscribed'])
    print(lr.score(np.expand_dims(df[i],axis=-1),df['term_deposit_subscribed']))
    # can choose all the cont_data as features since the accuracy > 0.89
    
# Categorical versus Categorical
for cat in cat_data:
    print(cat)
    confussion_mat = pd.crosstab(df[cat], df['term_deposit_subscribed']).to_numpy() 
    cv=CramersV()
    print(cv.cramers_corrected_stat(confussion_mat))
    # job type - 0.14
    # marital - 0.06
    # edu - 0.07
    # default - 0.02
    # h loan - 0.14
    # p loan - 0.07
    # com type - 0.15
    # month - 0.27
    # prev camp - 0.34  #can be considered
      
# Finalized features
# Removed prev_campaign_outcome because it doesnt cause any significant changes in the model accuracy

X=df.loc[:,['customer_age','balance','day_of_month','last_contact_duration',
            'num_contacts_in_campaign','num_contacts_prev_campaign']]

y=df.loc[:,'term_deposit_subscribed']

#%% Data Preprocessing

# We choose to scale the data using standard scaler because it follows normal distribution
ss=StandardScaler()
X=ss.fit_transform(X)

SS_FILE_PATH = os.path.join(os.getcwd(),'model','stdscaled_cs.pkl')
with open(SS_FILE_PATH, 'wb') as file:
    pickle.dump(ss,file)

ohe=OneHotEncoder(sparse=False)
y=ohe.fit_transform(np.expand_dims(y,axis=-1))

OHE_FILE_PATH = os.path.join(os.getcwd(),'model','ohe_cs.pkl')
with open(OHE_FILE_PATH,'wb') as file:
    pickle.dump(ohe,file)

# Train Test split
X_train,X_test,y_train,y_test=train_test_split(X,y,
                                               test_size=0.3,
                                               random_state=123)

#%% Model Development

nb_features=np.shape(X)[1:]
nb_class = len(np.unique(y))

mc=ModelCreation()
model = mc.two_layer_nn_model(nb_class,nb_features,drop_rate=0.2,l1_nodenum=32,l2_nodenum=32)

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])

#callbacks
tensorboard_callback=TensorBoard(log_dir=LOG_PATH)
# early_stopping_callback = EarlyStopping(monitor='loss',patience=50)



hist=model.fit(X_train,y_train,
               validation_data=(X_test,y_test),
               batch_size=64,
               epochs=20,
               callbacks=[tensorboard_callback])

# Accuracy = 90.22%

#%% Model Architecture

plot_model(model)

#%% Model Evaluation

me=ModelEvaluation()
me.eval_plot(hist)
    
me.model_eval(model, X_test, y_test, label=['0','1'])


#%% Model Saving

model.save(CS_MODEL_SAVE_PATH)


#%% Discussion
# =============================================================================
# # Problem Statement

# Provided marketing campaign dataset by bank, can we predict the outcome of the campaign?
# =============================================================================
# =============================================================================
# EXPLORATORY DATA ANALYSIS

# Questions:
#     1. What kind of data are we dealing with?
#        - The data has 18 variables with 17 being features and one target variable.
#        - The target variable for this dataset is 'term_deposit_subscribed'.
 
#     2. Do we have missing values?
#        - There are loads of missing values in the dataset
#        - Variables with NaNs are
#               * customer_age
#               * marital
#               * balance
#               * personal_loan
#               * last_contact_duration,
#               * num_contacts_in_campaign
#               * days_since_prev_campaign_contact
#         - Whole loads of them were originating from 'days_since_prev_campaign_contact'
#         - We can consider to drop t'days_since_prev_campaign_contact' for better accuracy

#     3. Do we have duplicated datas?
#        - None 
        
#     4. Do we have extreme values?
#        - We have but we are not doing anything due to the nature of the dataset
       
#     5. How to choose the features to make the best out of the provided data?
#        - Used Logistic Regression to select continuous features with more than 50% accuracy
#        - Selected all the continuous variables since allc the accuracies were more than 80% 
#        - Used Cramers'V to select categorical features with more than 50% accuracy as well
#        - But chose none of the categorical feature for model developement because the accuracies were low. Indications that the features relationship with the target variable. 
# =============================================================================
# =============================================================================
# MODEL DEVELOPMENT & MODEL EVALUATION

#   - Built two layer models due to data's density
#   - increasing the number of nodes doesnt change the model accuracy
#   - Hence, changing number of nodes from 32 to doesnt indicate any significant difference to our model accuracy
#   - Increasing the number of epochs from 10,20 50 and 100 doesnt change the accuracy as well
#   - Hence, our model is finalized by using simple layer nn, 32 hidden nodes, 20 epochs by using 6 features (all continuous variables)
#   - model with early callbacks works almost the same as the model without earlycallbacks

# =============================================================================


