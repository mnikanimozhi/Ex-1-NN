<H3>ENTER YOUR NAME: KANIMOZHI K N</H3>
<H3>ENTER YOUR REGISTER NO. 212225230126</H3>
<H3>EX. NO.1</H3>
<H3>DATE 04.08.2026</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1: Importing the Libraries: Import the required Python libraries needed for data analysis and machine learning.<BR>

STEP 2: Importing the Dataset: Load the dataset into the program for processing and analysis.<BR>

STEP 3: Taking Care of Missing Data: Handle missing values by removing or replacing them with suitable values.<BR>

STEP 4: Encoding Categorical Data: Convert categorical (text) data into numerical format for machine learning models.<BR>

STEP 5: Normalizing the Data: Scale the features to a common range to improve model performance.<BR>

STEP 6: Splitting the Data into Training and Testing Sets: Divide the dataset into training and testing sets to train and evaluate the model.<BR>


##  PROGRAM:
```
# Importing Libraries
import io
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
df=pd.read_csv("Churn_Modelling.csv",index_col="RowNumber")         # Read the dataset from drive
df.head()

 # Finding Missing Values
df.isnull().sum()

 # Check For Duplicates
print(df.duplicated().sum())

# Remove Unnecessary Columns
df=df.drop(['Surname', 'Geography','Gender'], axis=1) 
# Normalize the dataset
scaler=StandardScaler()                                
df=pd.DataFrame(scaler.fit_transform(df))
df.head()

# Split the dataset into input and output
X,Y=df.iloc[:,:-1].values ,df.iloc[:,-1].values  

 # Splitting the data for training & Testing
print('Input:\n',X,'\nOutput:\n',Y) 
Xtrain,Xtest,Ytrain,Ytest = train_test_split(X, Y, test_size=0.2)  

# X Train and Test
print("Xtrain:\n" ,Xtrain, "\nXtest:\n", Xtest)    

# Y Train and Test
print("\nYtrain:\n" ,Ytrain, "\nYtest:\n", Ytest)                   


```


## OUTPUT:
Dataset:

![alt text](img1.png)


Missing values:

![alt text](img2.png)


Duplicates:

![alt text](img3.png)


Standardized data:

![alt text](img4.png)


Splitting:

![alt text](img5.png)

## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


