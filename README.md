<a><img alt = 'image' src="https://img.shields.io/badge/Spyder%20Ide-FF0000?style=for-the-badge&logo=spyder%20ide&logoColor=white%22%3E%3C/a%3E" ><a>
<a><img alt = 'image' src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white%22%3E%3C/a%3E" ><a>
<a><img alt = 'image' src="https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white" ><a>
<a><img alt = 'image' src="https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white" ><a>


# Deep_Learning_Customer_Segmentation
 Developed a deep learning model to predict the outcome of the campaign 

 
 ## Problem Statement
 ### Provided marketing campaign dataset by bank, can we predict the outcome of the campaign?

 ## Exploratory Data Analysis

### Questions:
     1. What kind of data are we dealing with?
        - The data has 18 variables with 17 being features and one target variable.
        - The target variable for this dataset is 'term_deposit_subscribed'.
 
 
     2. Do we have missing values?
        - There are loads of missing values in the dataset
        - Variables with NaNs are
               * customer_age
               * marital
               * balance
               * personal_loan
               * last_contact_duration,
               * num_contacts_in_campaign
               * days_since_prev_campaign_contact
         - Whole loads of them were originating from 'days_since_prev_campaign_contact'
         - We can consider to drop t'days_since_prev_campaign_contact' for better accuracy
![msno](https://user-images.githubusercontent.com/105897390/175023464-ef62a19a-e824-4b44-9903-f49c46f95845.png)
*There are a lot of missing values for 'days_since_prev_campaign_contact' We can consider this variable for better accuracy.*

 
     3. Do we have duplicated datas?
        - None 
        
     4. Do we have extreme values?
        - We have but we are not doing anything due to the nature of the dataset
       
     5. How to choose the features to make the best out of the provided data?
        - Used Logistic Regression to select continuous features with more than 50% accuracy
        - Selected all the continuous variables since allc the accuracies were more than 80% 
        - Used Cramers'V to select categorical features with more than 50% accuracy as well
        - But chose none of the categorical feature for model developement because the accuracies were low. Indications that the features relationship with the target
 variable. 

### Model Development & Evaluation

   - Built two layer models due to data's density
   - increasing the number of nodes doesnt change the model accuracy
   - Hence, changing number of nodes from 32 to doesnt indicate any significant difference to our model accuracy
   - Increasing the number of epochs from 10,20 50 and 100 doesnt change the accuracy as well
   - Hence, our model is finalized by using simple layer nn, 32 hidden nodes, 20 epochs by using 6 features (all continuous variables)
   - model with early callbacks works almost the same as the model without early callbacks

 ![model_architecture](https://user-images.githubusercontent.com/105897390/175024065-9a0c3f7c-3f66-4611-8540-563cdee711d6.png)

 *This is the two layered neural network model developed this dataset in order to predict the outcome of the campaign.
 
![tensor_final](https://user-images.githubusercontent.com/105897390/175023808-aeb939b0-7dee-481f-9132-f9cb9ab62da1.png)

 *Attached is the image of tensorboard for the models I have trained using different number of hidden layers, hidden nodes, epochs and with early callbacks.
 
 
 ![confusion_matrix](https://user-images.githubusercontent.com/105897390/175025637-c2094705-ffac-4517-a5ea-78b2b3251809.png)
 
 *This is the confusion matrix produced by the model that we have developed.
 
![model_eval](https://user-images.githubusercontent.com/105897390/175024267-5660a974-3938-43b7-a26b-a58d26989476.png)
 
 *This is our model's accuracy when tested using test_dataset.
 
Accuracy - Accuracy is the most intuitive performance measure and it is simply a ratio of correctly predicted observation to the total observations. For our model, we have got 0.90 which means our model is approximately 90% accurate.
 
Accuracy = TP+TN/TP+FP+FN+TN
 
Precision is the ratio of correctly predicted positive observations to the total predicted positive observations. We have got 0.91 (91%) precision which is pretty good.
 
Precision = TP/TP+FP

Recall (Sensitivity) - Recall is the ratio of correctly predicted positive observations to the all observations in actual class. We have got recall of 0.99 which is good.

Recall = TP/TP+FN
 
F1 score - F1 Score is the weighted average of Precision and Recall. F1 is usually more useful than accuracy, especially if we have an uneven class distribution. In our case, F1 score is 0.95.

F1 Score = 2*(Recall * Precision) / (Recall + Precision)
