## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
       # INCLUDE YOUR CODING AND OUTPUT SCREENSHOTS HERE
```
import pandas as pd
df=pd.read_csv("C:\\Users\\admin\\Downloads\\data.csv")
df
```
<img width="639" height="506" alt="image" src="https://github.com/user-attachments/assets/f1a49ffc-8ee9-4fd8-9a6f-48ce85e45c4f" />

```
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder
df1=df.copy()
edu=["High School","Diploma","Bachelors","Masters","PhD"]
enc=OrdinalEncoder(categories=[edu])
enc.fit_transform(df1[['Ord_2']])
```
<img width="611" height="389" alt="image" src="https://github.com/user-attachments/assets/559c36ca-f991-44fc-aec4-120db265c761" />

```
df1['ordinalencoder']=enc.fit_transform(df1[['Ord_2']])
df1
```
<img width="710" height="474" alt="image" src="https://github.com/user-attachments/assets/a37c98f2-bf93-425b-bb4a-d93e6b36deac" />
```
df2=df.copy()
enc=LabelEncoder()
df2['LabelEncoder']=enc.fit_transform(df2[['Ord_2']])
df2
```
<img width="930" height="554" alt="image" src="https://github.com/user-attachments/assets/cbbbb840-3041-4b7d-9a0e-88d4367d15fb" />

```
from sklearn.preprocessing import OneHotEncoder
df3=df.copy()
enc=OneHotEncoder()
new=pd.DataFrame(enc.fit_transform(df3[['City']]))
df4=pd.concat([df3,new],axis=1)
df4
```
<img width="599" height="541" alt="image" src="https://github.com/user-attachments/assets/4100d6ac-08d1-4d0e-80ac-36c0578e69db" />

```
pd.get_dummies(df4,columns=['City'])
```
<img width="986" height="409" alt="image" src="https://github.com/user-attachments/assets/13e063cc-0179-4bfd-a524-0aca38f819fb" />

```
from category_encoders import BinaryEncoder,TargetEncoder
df5=df.copy()
enc=BinaryEncoder()
new=pd.DataFrame(enc.fit_transform(df5[['Ord_1']]))
df6=pd.concat([df5,new],axis=1)
df6
```
<img width="733" height="532" alt="image" src="https://github.com/user-attachments/assets/0e1cdb0f-d536-4331-b66d-7b3b743aca63" />
```
df7=df.copy()
enc=TargetEncoder()
new=pd.DataFrame(enc.fit_transform(df[['Ord_1']],df['Target']))
df8=pd.concat([df7,new],axis=1)
df8
```
<img width="625" height="482" alt="image" src="https://github.com/user-attachments/assets/b8420350-4b22-4b03-8fea-9f2a76e6a35b" />

```
df=pd.read_csv("C:\\Users\\admin\\Downloads\\Data_to_Transform.csv")
df
```
<img width="811" height="510" alt="image" src="https://github.com/user-attachments/assets/33994b63-d4b3-4595-9185-8beaa656961a" />

```
df.skew()
```
<img width="404" height="167" alt="image" src="https://github.com/user-attachments/assets/3a4e1b53-68fa-43fe-8356-f2aa55d10299" />

```
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
```
```
sm.qqplot(df["Moderate Positive Skew"],line="45")
plt.show()
```
<img width="861" height="616" alt="image" src="https://github.com/user-attachments/assets/145482bb-f6e2-4e4b-90ee-24b31172ae1e" />
```
sm.qqplot(df["Highly Positive Skew"],line="45")
plt.show()
```
<img width="785" height="600" alt="image" src="https://github.com/user-attachments/assets/fe2042ad-cf54-4f0d-9589-c7d25d0c81d0" />
```
sm.qqplot(df["Highly Negative Skew"],line="45")
plt.show()
```
<img width="753" height="604" alt="image" src="https://github.com/user-attachments/assets/7c748ba2-f7cf-44bb-8e48-71a79964ebbf" />
```
sm.qqplot(df["Moderate Negative Skew"],line="45")
plt.show()
```
<img width="812" height="609" alt="image" src="https://github.com/user-attachments/assets/076f428f-5f08-4664-bb83-7f68f4a7dec1" />
```
df1=df.copy()
df1['log transformation']=np.log(df["Moderate Positive Skew"])
df1
```
<img width="952" height="538" alt="image" src="https://github.com/user-attachments/assets/eea13d11-2587-4da8-98f5-66f84f29f85c" />
```
sm.qqplot(df1['log transformation'],line="45")
plt.show()
```
<img width="778" height="592" alt="image" src="https://github.com/user-attachments/assets/6effa22c-2c6a-4f2a-96cf-c9c01190e04e" />

```
df2=df1.copy()
df2['square root transformation']=np.sqrt(df1["Moderate Positive Skew"])
df2
```
<img width="1144" height="499" alt="image" src="https://github.com/user-attachments/assets/220f2cc0-d1ba-410f-ae51-3ec3ed74cc80" />

```
sm.qqplot(df2['square root transformation'],line="45")
plt.show()
```
<img width="760" height="614" alt="image" src="https://github.com/user-attachments/assets/8f407010-b9e0-43db-a138-29fbfb83838b" />

```
df3=df2.copy()
df3['square transformation']=np.square(df2["Highly Positive Skew"])
df3
```
<img width="1258" height="554" alt="image" src="https://github.com/user-attachments/assets/cceefd25-5d81-41f5-89c2-e4706b991c03" />

```
sm.qqplot(df3['square transformation'],line="45")
plt.show()
```
<img width="728" height="618" alt="image" src="https://github.com/user-attachments/assets/418b234b-50f8-47a0-ac7f-c0d7d9e476fb" />

```
df3=df2.copy()
df3['square transformation']=np.square(df2["Moderate Positive Skew"])
df3
```
<img width="1256" height="524" alt="image" src="https://github.com/user-attachments/assets/2ff6f0b2-2788-4d5f-9076-1d945507f157" />

```
sm.qqplot(df3['square transformation'],line="45")
plt.show()
```
<img width="849" height="612" alt="image" src="https://github.com/user-attachments/assets/9793ce70-391e-455d-ad30-fe5a1733ddcb" />

```
df4=df.copy()
df4['reciprocal transformation']=1/(df4["Moderate Positive Skew"])
df4
```
<img width="1126" height="503" alt="image" src="https://github.com/user-attachments/assets/983eaff8-d9ad-487e-8f64-500bff693192" />

```
sm.qqplot(df4['reciprocal transformation'],line="45")
plt.show()
```
<img width="789" height="626" alt="image" src="https://github.com/user-attachments/assets/f1a5a3ac-0e33-43c6-b1a3-4b48c10f8526" />

```
df5=df.copy()
df5['boxcox transformation'],p=stats.boxcox(df5["Moderate Positive Skew"])
df5
```
<img width="1178" height="552" alt="image" src="https://github.com/user-attachments/assets/1f237d49-9c2e-4523-b462-5f9147af1e77" />

```
sm.qqplot(df5['boxcox transformation'],line="45")
plt.show()
```
<img width="779" height="609" alt="image" src="https://github.com/user-attachments/assets/f5e6f662-5c36-4157-bfd7-80fa37cd6646" />

```
df6=df.copy()
df6['yeojohnson transformation'],p=stats.yeojohnson(df6["Moderate Negative Skew"])
df6
```
<img width="1036" height="497" alt="image" src="https://github.com/user-attachments/assets/33cb8ff0-7d6c-4a14-85d7-97d09182c13b" />

```
sm.qqplot(df6['yeojohnson transformation'],line="45")
plt.show()
```
<img width="789" height="607" alt="image" src="https://github.com/user-attachments/assets/f95aa149-aaab-4491-8355-def04a858b17" />

```
from sklearn.preprocessing import QuantileTransformer
df7=df.copy()
qt=QuantileTransformer(output_distribution='normal')
df7['Quantile Transformation']=qt.fit_transform(df7[['Highly Positive Skew']])
df7
```
<img width="1021" height="570" alt="image" src="https://github.com/user-attachments/assets/50d772e6-f0cb-4304-9736-045b2be00b28" />

```
sm.qqplot(df7['Quantile Transformation'],line="45")
plt.show()
```
<img width="724" height="606" alt="image" src="https://github.com/user-attachments/assets/d310f219-0543-433e-9b3e-11afbeabf358" />



# RESULT:
       # INCLUDE YOUR RESULT HERE
       <img width="724" height="606" alt="image" src="https://github.com/user-attachments/assets/ecc4c76b-67c1-406f-b0ce-4037a0a1ed26" />


       
