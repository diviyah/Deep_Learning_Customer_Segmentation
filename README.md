<a><img alt = 'image' src="https://img.shields.io/badge/Spyder%20Ide-FF0000?style=for-the-badge&logo=spyder%20ide&logoColor=white%22%3E%3C/a%3E" ><a>
<a><img alt = 'image' src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white%22%3E%3C/a%3E" ><a>
<a><img alt = 'image' src="https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white" ><a>
<a><img alt = 'image' src="https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white" ><a>
<a><img alt = 'image' src="(https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white" ><a>

# Deep_Learning_Customer_Segmentation
 Developed a deep learning model to predict the outcome of the campaign 

 
 # # Problem Statement

# Provided marketing campaign dataset by bank, can we predict the outcome of the campaign?

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

