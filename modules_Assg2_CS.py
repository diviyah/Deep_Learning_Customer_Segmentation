# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 18:49:58 2022

@author: ASUS
"""

import numpy as np
import seaborn as sns
import scipy.stats as ss
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input,Dense,Dropout,BatchNormalization
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import ConfusionMatrixDisplay

class EDA():
    def __init__(self): 
        pass
    
    
    def cont_plot(self,cont_data,df):
        '''
        Generate distribution plot for continuuous variables in the dataset.

        Parameters
        ----------
        cont_data : list
            List of numeric features.

        Returns
        -------
        Distribution plots of columns names provided in argument.

        '''
        for i in cont_data:
            plt.figure()
            sns.distplot(df[i])
            plt.show()



    def cat_plot(self,cat_data,df):
        '''
        Generate count plot for categorical features in a dataset.

        Parameters
        ----------
        cat_data : list
            List of categorical features.

        Returns
        -------
        Count plots of columns names provided in argument.

        '''
        for i in cat_data:
            plt.figure(figsize=(10,6))
            sns.countplot(df[i])
            plt.show()


    def cat_hue_plot(self,cat_data,df):
        '''
        

        Parameters
        ----------
        cat_data : list
            List of categorical features.

        Returns
        -------
        Count plots of categorical data's relationship with targer variable

        '''
        
        for i in cat_data:
            plt.figure()
            sns.countplot(df[i],hue=df['term_deposit_subscribed'])
            plt.show()
            
            
    def groupby_plot(self,cat_data,df):
        '''
        

        Parameters
        ----------
        cat_data : list
            List of categorical features.

        Returns
        -------
        groupby plot to get earlier inference of the data before analysis

        '''
        
        
        for cat in cat_data:
            df.groupby([cat,'term_deposit_subscribed']).agg({'term_deposit_subscribed':'count'}).plot(kind='bar',figsize=(10,5))



class CramersV():
    def __init__(self):
        pass
    
    def cramers_corrected_stat(self,confusion_matrix):
        """ calculate Cramers V statistic for categorial-categorial association.
            uses correction from Bergsma and Wicher, 
            Journal of the Korean Statistical Society 42 (2013): 323-328
        """
        chi2 = ss.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum()
        phi2 = chi2/n
        r,k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))
    


class ModelCreation():
    def __init__(self):
        pass
    
    
    def two_layer_nn_model(self,nb_class,nb_features,drop_rate=0.2,l1_nodenum=32,l2_nodenum=32):
           
        # activation function is linear just because our balance data set has negative values
        # chose output layer's activation function as sigmoid because the data is only binary classification problem
        model=Sequential()
        model.add(Input(shape=nb_features)) 
        model.add(Dense(l1_nodenum,activation='linear',name='Hidden_Layer1'))
        model.add(BatchNormalization())
        model.add(Dropout(drop_rate))
        model.add(Dense(l2_nodenum,activation='linear',name='Hidden_Layer2')) 
        model.add(BatchNormalization())
        model.add(Dropout(drop_rate))
        model.add(Dense(nb_class,activation='sigmoid',name='Output_Layer')) 
        
        model.summary()
        
        return model



class ModelEvaluation():
    def __init__(self):
        pass
    
    
    def eval_plot(self,hist):
        '''
        Generate graphs to evaluate model

        Parameters
        ----------
        hist : model
            Model fitted with train and test dataset.

        Returns
        -------
        Returns plots of loss and metrics that is assigned in model.compile()

        '''
        temp=[]
        
        for i in hist.history.keys():
            temp.append(i)

        for i in temp:
            if 'val_' in i:
                break
            else:
                plt.figure()
                plt.plot(hist.history[i])
                plt.plot(hist.history['val_'+i])
                plt.legend([i,'val_'+i])
                plt.show()
                
                
    def model_eval(self,model,X_test,y_test,label):
        '''
        Generates confusion matrix and classification report based
        on predictions made by model using test dataset.

        Parameters
        ----------
        model : model
            Prediction model.
        x_test : ndarray
            Columns of test features.
        y_test : ndarray
            Target column of test dataset. 
        label : list
            Confusion matrix labels.

        Returns
        -------
        Returns numeric report of model.evaluate(), 
        classification report and confusion matrix.

        '''
        result = model.evaluate(X_test,y_test)
        print(result) # loss and acc metrics
        y_pred=np.argmax(model.predict(X_test),axis=1)
        y_true=np.argmax(y_test,axis=1)
        print(y_true)
        print(y_pred)
        
        cm=confusion_matrix(y_true,y_pred)
        cr=classification_report(y_true, y_pred)
        print(cm)
        print(cr)
        
        disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=label)
        disp.plot(cmap=plt.cm.Reds)
        plt.show()

    

        
                


