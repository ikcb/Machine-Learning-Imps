MIT License

Copyright (c) 2021 Kunal Verma

Permission is hereby granted, free of charge, to any person obtaining a copy
of this model and associated documentation files , to deal
in the development of classification models without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the files.

THE FILE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


<==============================Explanation=============================>


I. <===============SOURCE FILES=======================================>

1. main.ipynb - Jupyter Notebook. Ran on my local kernel
2. Training Data.csv - Training Data that has been used to train the model
3. Test Data.csv - Test Data that has been used to evaluate the model
4. Submission.csv - This file is the labels that the model predicted for the Test Data
5. requirements.txt - This file includes all libraries and frameworks that has been used to train the model.


II. <=================Tools Used======================>

I. Python Programming Language - version==3.7.6

II. Python Libraries and Frameworks - 

1. Numpy - version==1.18.1
2. Matplotlib - version==3.1.3
3. Tensorflow - version==2.2.0
4. sklearn - version==0.23.2

III. <==================Approach===============>

1. First thing data is quite resembles the churdataset so I just gave it a first try 
and achieved 88.5714 % on test set that has been evaluated on Tech Gig Evaluation Process.

2. Dataset has includes 56 columns and 965 rows in Trainning Data.csv file

The columns that are present in are : 

clientid_cr
clmbuserid_cr
conversiontime_cr
imprid_cr
adslotdimid_cr
algo_cr
audiences_cr
clickbid_cr
geodimid_cr
ip_cr
itemcolumbiaid_cr
itemid_cr
position_cr
pubclientid_cr
refurl_cr
siteId_cr
templateid_cr
goalid_cr
time_cr
adLogType_cr
v_cr
allAudiences_cr
pricingtype_cr
osId_cr
browserId_cr
cityId_cr
stateId_cr
modelDimId_cr
lookUpFrom_cr
connTypeDimId_cr
ispDimId_cr
countryDimId_cr
goalTypeId_cr
conversionDurationInMillis_cr
impressionTimeInMillis_cr
clickTimeInMillis_cr
osVerDimId_cr
uuidSource_cr
geoGrpDimId_cr
stateGrpDimId_cr
deviceId_cr
uvh_cr
uv_cr
platformId_cr
sdkVersion_cr
usrClusterId_cr
cityGrpDimId_cr
siteClusterIds_cr
refClusterId_cr
paid_cr
spend_cr
attributionType_cr
conversionid_cr
optimize_on_cr
bundleId_cr
conversion_fraud

3. But there are some redundant columns that are not required as I tested it before training the data. The columns which includes user id and record id and etc, do not affect the performance of the model as these are values that are randomly given to the users.

The columns that I used are - 

clientid_cr
conversiontime_cr
adslotdimid_cr
algo_cr
clickbid_cr
geodimid_cr
itemcolumbiaid_cr
itemid_cr
position_cr
pubclientid_cr
siteId_cr
templateid_cr
goalid_cr
time_cr
adLogType_cr
v_cr
pricingtype_cr
osId_cr
browserId_cr
cityId_cr
stateId_cr
modelDimId_cr
lookUpFrom_cr
connTypeDimId_cr
ispDimId_cr
countryDimId_cr
goalTypeId_cr
conversionDurationInMillis_cr
impressionTimeInMillis_cr
clickTimeInMillis_cr
osVerDimId_cr
uuidSource_cr
geoGrpDimId_cr
stateGrpDimId_cr
deviceId_cr
uvh_cr
uv_cr
platformId_cr
usrClusterId_cr
cityGrpDimId_cr
siteClusterIds_cr
refClusterId_cr
paid_cr
spend_cr
attributionType_cr
optimize_on_cr
conversion_fraud

IV. <===============Feature Engineering======>

1. Dealing with the numbers that have redundant character like "," ".", etc.

To deal with these unwanted character and to preserve the data, I used python concept that first replace these character with empty string and then converts them to float values so as to use in the dataset.

To achieve above below is the Pseudo Code - 

data = pandas.read_csv('training data.csv')

data[column_name]=data[column_name].str.replace(",","").astype(float);


5. Dealing with Missing Values in the Dataset

To achieve we can use the technique in Data Science which is called imputation. 
This technique states that the missing values in any dataset can be converted to the mean of all values in that particular column.

Scikit Learn provides the in-built class to do this tedious task.

Create an instance of impute class and fit it to the dataset

Below is the Pseudo Code - 
imputer = SimpleImputer(missing_values=np.nan, strategy='mean', verbose=0)

X = imputer.fit_transform(X)


6. Feature Scaling

As the values in the dataset ranges from 1000s to 1000000s so they are not in a similar range.
So to train our model we need to make them in same range lets say between 0 to 1.

This can be done from StandarScaler() class provided by Scikit Learn .

Below is the Pseudo Code - 

fs = StandardScaler()
X = fs.fit_transform(X)

7. Label Encoding 

Since our test labels has true and false so we need to convert it to 0 and 1 so as to feed the model

V. <===============Making the Model=========>

I have used here the state-of-the-art, Random Forest Classifier with 300 trees in the forest.

After training the model on our dataset, we predict the model on test set and it evaluates around 88.424%.


