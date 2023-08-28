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

Here is a snapshot of what the dataset looks like:    
![image](https://github.com/RLButch/project-4-stroke-predictions/assets/122842203/8edb0f0f-a3a8-4269-8825-ce6882a4e195)


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
Dataset from kaggle was uploaded, cleaned and organised in order to get it ready for machine learning and storage in pickle before finally uploading into flask for visualisation of our project.  
Firstly we uploaded the csv file into python using pandas, visualised the data into a dataframe and then explored some key features of the dataset, we looked at the number of stroke incidence vs no stroke.  
We explored the dataset learn what values are numerical and non numerical, what columns exist in the data, what columns might be missing data - we can see the bmi column is missing some values (approx. 201).  
We performed some basic summary statistics on the dataset to get an idea of the values and characteristics.  
We investigated columns (or rather 'features' such as bmi, age, gender etc) that didn't have values (nan). we found 201 data entries missing for the BMI column. 
We removed any duplicates and the 'id' column, as this column doesn't add any value to the further investigation of the dataset. 
We performed a correlation matrix between the numerical features in the data. (results explained further on).  
Outliers were removed from the dataset. (explaination further on).  we examined the shape of the data before and after removing the outliers:
Shape of data before removing outliers : (5110, 11)  
Shape of data after remove : (4401, 11)   
We then grouped BMI by age group and then in order to retain data records the bmi column with NAN values was replaced with the average for that age group.  The machine learning algorithm cannot process categorical data values. It can only process numerical values. 
To fit the data into the prediction model, we need to convert categorical values to numerical ones. Before that, we will evaluate if any transformation on categorical columns is necessary. 
We visualised the distribution of the data: checking the numerical (columns in the dataset that value numbers rather than letters) key features of the dataset, interesting to see that bmi is almost normally distributed  
![image](https://github.com/RLButch/project-4-stroke-predictions/assets/122842203/5869141b-c2ed-4459-8e7e-294eb8349fd8)   
We then checked the gender distribution, since the 'other' gender only had one entry, it was decided to remove this and the removal would not adversly affect the data or subsequent machine learning models.  
We then decided to investigate the non-numerical dataset features, looking at the gender distribution, job types held (work counts), distribution of smokers vs non smokers etc and the residence types. This was to get a good understanding of what the data looks like and some key non-numerical features held within the dataset.`
![image](https://github.com/RLButch/project-4-stroke-predictions/assets/122842203/492714a9-8436-4955-8da6-5bcdab732d72)  

We then investigated a breakdown of the dataset non-numerical features by stroke incidence. its interesting to see that smoking didn't seem to have a large impact on the incidence of stroke where people who have never smoked had a slightly higher incidence of stroke than those that smoked. Hypertension didn't seem to have a great incidence neither did heart disease. Whether people lived rurally or in urban areas didn't seem to impact stroke incidence either. It seems there is a slightly higher incidence of stroke in women over men. it appears marriage leads to higher stroke incidence, which is in contrast to a high number of successful research into the benefits of marriage which includes improved health outcomes. This dataset seems to buck the trend.   
![image](https://github.com/RLButch/project-4-stroke-predictions/assets/122842203/d3bfd06c-764e-4564-a2e1-33e85c6ee672)   

**Data preprocessing** methods included:  
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
After much researching it was found that the random forest algorithm is idea for dealing with data imbalance. It is a strong modelling technique and is much sturdier than a single decision tree. The aggregation of several trees limits the possibility of data overfitting and miscalculations due to bias.  Random forest has also historically been a model of choice for healthcare datasets. In Random Forest, feature importance comes for free when training a model, so it is a great way to verify initial hypotheses and identify ‘what’ the model is learning.

Our first algorithm attempt was to use SMOTE (Synthetic Minority Oversampling Technique) with the random forest which resulted in a reasonable accuracy but wasn't predicting the data as efficiently as we would like. SMOTE is an oversampling technique that uses a minority class to generate synthetic samples. It typically overcomes overfitting problems raised by random oversampling. It randomly selects a minority case instance (in this case a stroke) and finds its nearest neighbour. Then it generates synthetic models by randomly choosing one of the neighbours and forms a line segement in the feature space. In this case of this project, it resulted in the least accuracy and lowest confusion matrix values (shown below). "Over-sampling does not increase information; however by replication it raises the weight of the minority samples"  (https://statistics.berkeley.edu/sites/default/files/tech-reports/666.pdf). It is important in health datasets that we don't over predict strokes but under prediction would possibly be a bigger problem - where an at risk patient goes undetected. SMOTE is not a perfect substitute for real data because it generates synthetic samples based on existing data, which can potentially introduce biases and patterns that might not accurately reflect the true distribution and intricacies of the underlying data. Additionally, SMOTE assumes that all features are equally important and independent, which may not hold true in many real-world scenarios where complex relationships exist among variables. Moreover, the process of oversampling through SMOTE can lead to overfitting, as it magnifies the noise present in the original dataset, potentially causing the model's performance to degrade when exposed to genuinely unseen data. Lastly, SMOTE doesn't consider the contextual nuances and domain-specific intricacies that real data often encapsulate, making it challenging to replicate the full spectrum of scenarios that a model might encounter in practical applications.
	
![image](https://github.com/RLButch/project-4-stroke-predictions/assets/122842203/027b2455-8e91-4a24-bf94-16cb8d7271cd)



![image](https://github.com/RLButch/project-4-stroke-predictions/assets/122842203/0c357154-87ae-4878-9743-3d9d0c4fe215)


Evaluation: In this optimization attempt, we utilized SMOTE to oversample the data, resulting in an overall accuracy score of approximately 98% when predicting class 0 (no stroke) and class 1 (stroke. For our next attempt, we will explore the binning method to see if this can help simplify the data and reduce noise. The objective is to optimize the model's overall performance, as well as enhance the sensitivity and specificity scores. The test data showed that it was predicting no stroke when actually there were 27 incidents of stroke, indicating this isn't the best model to be using for our final visualisation or the best model overall for predicting the likelihood of stroke. The F1 score is close to 1 which tells us that this model is making the correct predictions most of the time but its not quite there. Precision tells us that out of all the patients the model predicted no stroke - only 97% didn't have a stroke however for the incidence of stroke it was 100%. this model is predicting  stroke incidence better than it is predicting no strokes. 100% of the people that had a stroke it predicted it correctly but only predicted 97% of the people who didn't have a stroke.


Our second attempt included binning the BMI feature of the dataset and using oversampler and the random forest algorithm which produced better results than the first model.  Binning algorithms can pre-process the data and speed up the subsequent tree construction in a random forest. It is a pre-processsing method to group numerical values and is a technique that may address prevalent data issues such as the handling of missing values, presence of outliers and statistical noise as well as data scaling. 
Results from this second model were reasonable however upon further research, perhaps binning is unnecessary since when building a decision tree, the algorithm in a way, does the binning for you, based on the best split. so we decided to base the next model optimising attempt on just the random oversampler and random forest classifier. 
	
		
![image](https://github.com/RLButch/project-4-stroke-predictions/assets/122842203/26613061-ae6a-43fa-932e-8185fd50699e)

![image](https://github.com/RLButch/project-4-stroke-predictions/assets/122842203/d2d1105a-3da9-45a7-8844-d9fd3956bffa)


Evaluation: In this optimization attempt, we divided the bmi column into four categories(or bins) and applied the RandomOverSampler technique to oversample the data and utilized it for training our model. As a result, we obtained an accuracy score of approximately 99.6% for predicting both class 0 and 1. When we compared this outcome with our initial model that didn't involve binning the bmi, we observed that the overall performance increased slightly by 1.5%, while sensitivity and specificity scores were enhanced. This suggests that binning the bmi column has an affect on the overall performance of the model. The F1 score is equal to 1 which tells us that this model is predicting stroke incidence accurately. Precision tells us that out of all the patients the model predicted no stroke - the model is gettin it 100% correct and for stroke prediction the incidence  it was 97%. this model is predicting non stroke incidence better than it is predicting strokes. On the test data it was accurately predicting no strokes when there weren't any and predicting 6 strokes when there were no actual strokes.


Our third attempt was the oversampler and random forest model alone which became our model of choice and the final model for the final visualisation of this project - the data was imputed through pickle and visualised in flask, html and css.		
							
![image](https://github.com/RLButch/project-4-stroke-predictions/assets/122842203/fef4a295-260f-413c-8573-18e7e5643501)

	
![image](https://github.com/RLButch/project-4-stroke-predictions/assets/122842203/04128e1c-bde1-439a-aadf-adad377ba69e)

Evaluation: We have accomplished an high overall accuracy rate of approximately 99.6% when predicting both class 0 and 1 using this model. The F1 score is equal to 1 which tells us that this model is predicting stroke incidence accurately. Precision tells us that out of all the patients the model predicted no stroke - the model is getting it 100% correct and for stroke prediction the incidence  it was 99%. this model is predicting non stroke incidence slightly better than it is predicting strokes. On the test data it was accurately predicting no strokes when there weren't any and predicting 7 strokes when there were no actual strokes. The weighted average of precision, recall, and F1-score, considering class imbalance, is 1.00, further confirming the model's exceptional performance. In summary, this classification report illustrates an extremely accurate and well-balanced model, with both precision and recall being very close to perfect, resulting in high F1-scores and overall accuracy. However it is accurately predicting strokes when there are strokes. There were no strokes that were not predicted in this dataset. Due to the slightly higher accuracy from the classification report - this model was chosen to be uploaded in pickle and visualised through a flask app.

*Note: After comparing the overall accuracy score, false positive rate and false negative rate, we've decided this is the best model(our final model!).

All of the model's performance on the imbalanced dataset, we used common metrics like the confusion matrix, precision, recall, f1-score and PRC (precision-recall Curve).

Precision is defined as equation.  It is used to measure the positive patterns that are correctly predicted from the total predicted patterns in a positive class. 𝑃𝑟𝑒𝑐𝑖𝑠𝑖𝑜𝑛= 𝑇𝑃 𝑇𝑃+𝐹𝑃  
Another useful metric is recall, which is used to measure how well the fraction of a positive class becomes correctly classified (Hossin and Sulaiman 2015). recall is defined as equation - 𝑅𝑒𝑐𝑎𝑙𝑙= 𝑇𝑃 𝑇𝑃+𝐹𝑁   
The F1-score is a measure of model performance that combines precision and recall into a single number. This metric represents the harmonic mean between recall and precision values as equation  
𝐹1𝑠𝑐𝑜𝑟𝑒= 2/ 1 𝑃𝑟𝑒𝑐𝑖𝑠𝑖𝑜𝑛+ 1 𝑅𝑒𝑐𝑎𝑙𝑙 =2∗𝑃𝑟𝑒𝑐𝑖𝑠𝑖𝑜𝑛∗𝑟𝑒𝑐𝑎𝑙𝑙 𝑃𝑟𝑒𝑐𝑜𝑠𝑖𝑜𝑛+𝑅𝑒𝑐𝑎𝑙𝑙

**For some extra work** we decided to apply a logistical regresssion model to the data which only resulted in 50% accuracy.  In this optimization attempt, the logistical regression method is not very accurate, even after balancing the dataset. however the harmonic mean of precision and recall is also high (close to 1), indicating a good balance between precision and recall.

**Decision Tree**	

We also applied a decision tree model to the data and received an accuracy of 0.9752066115702479  The F1 score shows the model is 98% accurately predicting when a patient will have a stroke. It is very accurate for stroke prediction but not as accurate as other models for non-stroke prediction. 

Classification Report
              precision    recall  f1-score   support

           0       0.95      1.00      0.97       815
           1       1.00      0.95      0.98       879

    accuracy                           0.98      1694
   macro avg       0.98      0.98      0.98      1694
weighted avg       0.98      0.98      0.98      1694

We also optimised this model using hyper parameters.  

params_grid = [{'max_depth': [2, 3, 4], 'min_samples_leaf': [10, 20, 30], 'max_features': [3, 5, 7]}]  
Get the best combination of parameters
grid_search_raw.best_params_
The results presented above reveal that both the training and test sets exhibit approximately equivalent class proportions.

Moving forward, the initial Decision Tree classifier can be trained using the unprocessed (unscaled) training dataset. Before fitting the model, we employed GridSearchCV with a 3-fold cross-validation strategy to identify the optimal combination of hyperparameter values for the classifier. The following three hyperparameters have been selected for grid search:

max_depth: This parameter determines the maximum depth of the tree. A higher value leads to a more intricate tree with increased flexibility, potentially resulting in overfitting.
min_samples_leaf: Setting this parameter establishes the minimum number of samples required in a leaf node. Opting for a smaller value might create numerous nodes with insufficient samples, contributing to overfitting.
max_features: This parameter specifies the maximum count of features considered for node splitting. The risk of overfitting increases if too many features are included.
For each of these hyperparameters, I will explore three different values to comprehensively assess their impact.

0.9752066115702479
Classification Report
              precision    recall  f1-score   support

           0       0.95      1.00      0.97       815
           1       1.00      0.95      0.98       879

    accuracy                           0.98      1694
   macro avg       0.98      0.98      0.98      1694
weighted avg       0.98      0.98      0.98      1694

GridSearchCV
estimator: DecisionTreeClassifier

DecisionTreeClassifier
Utilizing the trained tree model, we can predict outcomes for the test dataset and assess its efficacy on previously unseen information. Subsequently, we can gauge its effectiveness using a performance measure like the F1 score. Employing a three-fold cross-validation technique, We will appraise the model and obtain three distinct F1 scores as output.
We created a grid of paramaters to search, however were unable to optimise the model.
{'max_depth': 2, 'max_features': 5, 'min_samples_leaf': 20} were identified as the best parameters for the dataset
Cross-validation with three folds to get average F1 score  
Scores: [0.76106195 0.63362832 0.58510638]   
Mean of scores: 0.6599322161551497    
On two of the folds, this model achieved an F1 score over 60%, but only got 58% on the final fold. On average, this model's F1 score is 65%, which can definitely be improved upon.
![tree_model](https://github.com/RLButch/project-4-stroke-predictions/assets/122842203/3b8fd9ca-990d-4a1a-b4dd-7531bbccf368)  

**Pickle, flask and html deployment**		
Flask is a Python-based micro framework used for developing small-scale websites. We imported the libraries, then using app=Flask(__name__) we create an instance of flask. @app.route(‘/’) is used to tell flask what URL should trigger the function index() and in the function index, we use render_template(‘index.html’) to display the script index.html in the browser.This should run the application and launch a simple server. Open http://127.0.0.1:5000/ to see the html form.
HTML Form: We first need to collect the data(new attribute values) to predict stroke from various attributes and then use the random forest model we built above to predict the likelihood of stroke. Therefore, in order to collect the data, we create an HTML form which would contain all the different options to select from each feature.
Predicting strokes: When someone submits the form, the webpage should display the predicted incidence (or not) of stroke. For this, we require the model file (model.pkl) we created.
Here, after the form is submitted, the form values are stored in the variable to_predict_list in the form of a dictionary. We convert it into a list of the dictionary’s values and pass it as an argument to ValuePredictor() function. In this function, we load the model.pkl file and predict the new values and return the result. 
This result/prediction (stroke or no stroke) is then passed as an argument to the template engine with the HTML page to be displayed.
POST Requests:
The POST method is used to send data to the server to create or update a resource. It's often used when you want to submit forms, send data to the server for processing, or make changes on the server.
To handle POST requests in Flask, we needed to use the request object to access the data sent by the client. The GET method is used to retrieve data from the server. It's typically used when you want to retrieve information from the server and display it to the user, or when you want to retrieve data without making any changes on the server.
It uses HTML forms to gather user inputs related to various health parameters, processes the inputs, and then uses a pre-trained machine learning model to predict the risk of stroke based on the provided information.
The Flask app contains two routes:
The main route ("/") renders an HTML template called "index.html" on the homepage.
The "/output" route handles both POST and GET requests. It processes the user's inputs and attempts to make a prediction regarding the risk of stroke. The outcome of this prediction is then displayed on a result page template called "index 2.html".
The code uses the following key steps:
User inputs related to gender, age, health conditions, lifestyle factors, etc., are obtained from the submitted form. (data must be entered in each field)
These inputs are then processed and transformed to match the format expected by a pre-trained machine learning model for stroke prediction uploaded via pickle.
The machine learning model is loaded using the pickle library, and the processed inputs are used to make a prediction.
Depending on the prediction result, appropriate messages are generated and displayed to the user, indicating the risk of stroke.
In case of invalid inputs or errors in processing, relevant messages are displayed to the user.
It's important to note that certain functions such as stroke_pred and fancy_deconstruct are used to handle specific data transformation and prediction processes. The app uses render_template to dynamically display HTML pages and request.form to access form data submitted by users.
Finally, the app is run using app.run() only when the script is executed directly (not when it's imported as a module).
Keep in mind that this app.py relies on HTML templates ("index.html" and "index 2.html") to provide a user interface. The HTML templates contain the form fields and result display areas.


