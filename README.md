# project-4-stroke-predictions
Stroke Prediction Machine Learning Model  
![stroke-3233778_1280](https://github.com/RLButch/project-4-stroke-predictions/assets/122842203/26120b59-40df-4e71-94ef-acf465316a07)    

**Project Description**    

A stroke is a medical emergency. A stroke happens when blood cannot get to your brain, because of a blocked or burst artery. As a result, your brain cells die due to a lack of oxygen and nutrients.
Symptoms of stroke include trouble walking, speaking and understanding, as well as paralysis or numbness of the face, arm or leg.
Early treatment with medication like tPA (clot buster) can minimise brain damage. Other treatments focus on limiting complications and preventing additional strokes. Therefore intervention or understanding risk factors for stroke occurence is an important topic within the healthcare system. The objective of this project is to use machine learning to develop an early detection model for strokes. This will be based in causative factors determined during a deep dive into the dataset downloaded from Kaggle.

**Data Overview**    

The dataset used for stroke incidence was drawn from a compliation of patients medical records. It encompasses wide ranging information including patient medical histories, demographics, lifestyle factors and the presence or absence of a stroke for each patient.

Here is a snippet of the dataset:    


**Resources:**     

This project was built with:  
Python Packages (eg. scikit-learn, matplotlib, searborn, etc.)     
pickle     
Flask     
Javascript, D3      
HTML  
CSS  

**Approach:**    

Identify data sources and dependencies  
Perform EDA, determine feature set and transform stroke data  
Compile, train and evaluate the model  
Compare models for optimization of accuracy metric  
Store the transformed dataset into pickle
Create Flask App, import data in via pickle and connect routes to model  
Create interactive web app using pickle, html and css  
![image](https://github.com/RLButch/project-4-stroke-predictions/assets/122842203/2db5ce0a-f698-4507-8c17-350d492f08cc)
 

Statistical analysis was performed on the data: 

 Interquartile range (IQR)
 The interquartile range (IQR) is a measure of variability used in statistical analysis. It is a measure of the spread of a dataset and is defined as the difference between the upper and lower quartiles of the dataset. To calculate the IQR, the dataset is first sorted in ascending order. The median is then calculated, and the dataset is split into two halves - the lower half and the upper half. The lower quartile (Q1) is the median of the lower half, and the upper quartile (Q3) is the median of the upper half. The IQR is then calculated as the difference between Q3 and Q1. The IQR method is often used to identify outliers in a dataset. Any value that falls below Q1 - 1.5IQR or above Q3 + 1.5IQR is considered an outlier and may be removed from the dataset.

Unsupervised Machine Learning Methods







