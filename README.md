# project-4-stroke-predictions
Stroke Prediction Machine Learning Model  
![stroke-3233778_1280](https://github.com/RLButch/project-4-stroke-predictions/assets/122842203/26120b59-40df-4e71-94ef-acf465316a07)    

**Project Description**    

A stroke is a medical emergency. A stroke happens when blood cannot get to your brain, because of a blocked or burst artery. As a result, your brain cells die due to a lack of oxygen and nutrients.
Symptoms of stroke include trouble walking, speaking and understanding, as well as paralysis or numbness of the face, arm or leg.
Early treatment with medication like tPA (clot buster) can minimise brain damage. Other treatments focus on limiting complications and preventing additional strokes. Therefore intervention or understanding risk factors for stroke occurence is an important topic within the healthcare system. The objective of this project is to use machine learning to develop an early detection model for strokes. This will be based in causative factors determined during a deep dive into the dataset downloaded from Kaggle.
Using the Kaggle dataset we were able to achieve excellent accuracy of the dataset.

** Sources used in creation of this project**  
Kaggle, One on one tutorials, Wednesday Tutorials, Class activities, class instructor and TA assistance, stack overflow, medium, geek to geek websites, 

**Data Overview**    

The dataset used for stroke incidence was obtained from Kaggle and was drawn from a compliation of patients medical records. It encompasses wide ranging information including patient medical histories, demographics, lifestyle factors and the presence or absence of a stroke for each patient.

Here is a snippet of the dataset:    


**Resources:**     
This project was built with:  
Python Packages (eg. scikit-learn, pandas, matplotlib, seaborn, etc.)     
pickle     
Flask     
HTML  
CSS  

**Approach:**    

Identify data sources and import dependencies   
Perform EDA, determine feature set and transform stroke data   
Compile, train and evaluate the model   
Compare models for optimization of accuracy metric   
Store the transformed dataset into pickle 
Create Flask App, import data in via pickle and connect routes to model   
Create interactive web app using pickle, html and css   
![image](https://github.com/RLButch/project-4-stroke-predictions/assets/122842203/2db5ce0a-f698-4507-8c17-350d492f08cc)
 
**EDA** 
The kaggle dataset was cleaned 




Data preprocessing methods included:  
We applied oversampling methods to handle imbalanced data 
We encoded data to categorical variables (get dummies) 
We used feature scaling to transform numerical features into a consistent range (splitting into x and y, standardscaler) 
We divided the dataset data into training and testing sets using train_test_split  

Statistical analysis was performed on the data:  

 Interquartile range (IQR)
 The interquartile range (IQR) is a measure of variability used in statistical analysis. It is a measure of the spread of a dataset and is defined as the difference between the upper and lower quartiles of the dataset. To calculate the IQR, the dataset is first sorted in ascending order. The median is then calculated, and the dataset is split into two halves - the lower half and the upper half. The lower quartile (Q1) is the median of the lower half, and the upper quartile (Q3) is the median of the upper half. The IQR is then calculated as the difference between Q3 and Q1. The IQR method is often used to identify outliers in a dataset. Any value that falls below Q1 - 1.5IQR or above Q3 + 1.5IQR is considered an outlier and may be removed from the dataset.
![image](https://github.com/RLButch/project-4-stroke-predictions/assets/122842203/a6ed784b-35aa-4cea-8ca4-a38872169503)  
  
Outliers were removed from the dataset, BMI and avg glucose level fields contained quite a number of outliers and it was decided to clean the dataset of these, particularly as BMI ranging above 60 is very rare. (https://www.medicalnewstoday.com/articles/323446#waist-to-hip-ratio) During a search of the literature on BMI, there was very little mention of BMI ranges above 40. So it was appropriate to remove the outliers in the BMI field. (https://mexicobariatriccenter.com/morbidly-obese-chart-am-i-morbidly-obese/#:~:text=BMI%20%3D%2025%20to%2029.9%20%E2%80%93%20Overweight,%2B%20%E2%80%93%20Super%2DSuper%20Morbidly%20Obese) Over 60 on the BMI range is considered super super morbidly obese, these figures would indicate an already unhealthy individual who may skew the datasets and are not representative of the overall population. 
"The expected values for normal fasting blood glucose concentration are between 70 mg/dL (3.9 mmol/L) and 100 mg/dL (5.6 mmol/L). When fasting blood glucose is between 100 to 125 mg/dL (5.6 to 6.9 mmol/L) changes in lifestyle and monitoring glycemia are recommended". WHO, (https://www.who.int/data/gho/indicator-metadata-registry/imr-details/2380#:~:text=The%20expected%20values%20for%20normal,and%20monitoring%20glycemia%20are%20recommended.)  Therefore it was reasonable to exclude outliers from the average glucose levels field.


**Building Machine Learning models** 

![image (002)](https://github.com/RLButch/project-4-stroke-predictions/assets/122842203/c39f3349-413e-4c79-9b63-9994c1fee269)  
A correlation matrix was performed on the dataset between the numerical features in the data. As shown in the matrix, most of the features are not highly correlated with any other features. BMI has a weakly positive correlation with stroke which is the most obvious correlation within the matrix.  

A random forest model was chosen for this project as it fits the dataset - our dataset is slightly imbalanced, not unexpectedly but the lack of of stroke within the dataset is far lesser than the incidence of stroke which creates the imbalance in the dataset.  (it would be expected in this type of dataset that the non-stroke incidence would outweigh the stroke incidence, like in a finance dataset where standard transactions would outweigh the number of fraudulent transactions). 
After much researching it was found that the random forest algorithm is idea for dealing with data imbalance. It is a strong modelling technique and is much sturdier than a single decision tree. The aggregation of several trees limits the possibility of data overfitting and miscalculations due to bias.  Random forest has also historically been a model of choice for healthcare datasets.

Our first algorithm attempt was to use SMOTE (Synthetic Minority Oversampling Technique) with the random forest which resulted in a reasonable accuracy but wasn't predicting the data as efficiently as we would like. SMOTE is an oversampling technique that uses a minority class to generate synthetic samples. It typically overcomes overfitting problems raised by random oversampling. It randomly selects a minority case instance (in this case a stroke) and finds its nearest neighbour. Then it generates synthetic models by randomly choosing one of the neighbours and forms a line segement in the feature space. In this case of this project, it resulted in the least accuracy and lowest confusion matrix values (shown below). "Over-sampling does not increase information; however by replication it raises the weight of the minority samples"  (https://statistics.berkeley.edu/sites/default/files/tech-reports/666.pdf). It is important in health datasets that we don't over predict strokes but under prediction would possibly be a bigger problem - where an at risk patient goes undetected. Hence over-sampling is an acceptable technique on these types of datasets (imbalanced). 

Our second attempt included binning the BMI feature of the dataset and using oversampler and the random forest algorithm which produced better results than the first model.  Binning algorithms can pre-process the data and speed up the subsequent tree construction in a random forest. It is a pre-processsing method to group numerical values and is a technique that may address prevalent data issues such as the handling of missing values, presence of outliers and statistical noise as well as data scaling. 
Results from this second model were reasonable however upon further research, perhaps binning is unnecessary since when building a decision tree, the algorithm in a way, does the binning for you, based on the best split. so we decided to base the next model optimising attempt on just the random oversampler and random forest classifier. 

Our third attempt was the oversampler and random forest model alone which became our model of choice and the final model for the final visualisation of this project - the data was imputed through pickle and visualised in flask, html and css.

All of the model's performance on the imbalanced dataset, we used common metrics like the confusion matrix, precision, recall, f1-score and PRC (precision-recall Curve).

Precision is defined as equation.  It is used to measure the positive patterns that are correctly predicted from the total predicted patterns in a positive class. 𝑃𝑟𝑒𝑐𝑖𝑠𝑖𝑜𝑛= 𝑇𝑃 𝑇𝑃+𝐹𝑃  
Another useful metric is recall, which is used to measure how well the fraction of a positive class becomes correctly classified (Hossin and Sulaiman 2015). recall is defined as equation - 𝑅𝑒𝑐𝑎𝑙𝑙= 𝑇𝑃 𝑇𝑃+𝐹𝑁   
The F1-score is a measure of model performance that combines precision and recall into a single number. This metric represents the harmonic mean between recall and precision values as equation  
𝐹1𝑠𝑐𝑜𝑟𝑒= 2/ 1 𝑃𝑟𝑒𝑐𝑖𝑠𝑖𝑜𝑛+ 1 𝑅𝑒𝑐𝑎𝑙𝑙 =2∗𝑃𝑟𝑒𝑐𝑖𝑠𝑖𝑜𝑛∗𝑟𝑒𝑐𝑎𝑙𝑙 𝑃𝑟𝑒𝑐𝑜𝑠𝑖𝑜𝑛+𝑅𝑒𝑐𝑎𝑙𝑙









