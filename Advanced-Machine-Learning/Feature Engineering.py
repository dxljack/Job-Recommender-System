#!/usr/bin/env python
# coding: utf-8

# This is the last assignment for the Coursera course "Advanced Machine Learning and Signal Processing"
# 
# Just execute all cells one after the other and you are done - just note that in the last one you should update your email address (the one you've used for coursera) and obtain a submission token, you get this from the programming assignment directly on coursera.
# 
# Please fill in the sections labelled with "###YOUR_CODE_GOES_HERE###"
# 
# The purpose of this assignment is to learn how feature engineering boosts model performance. You will apply Discrete Fourier Transformation on the accelerometer sensor time series and therefore transforming the dataset from the time to the frequency domain. 
# 
# After that, you’ll use a classification algorithm of your choice to create a model and submit the new predictions to the grader. Done.
# 
# Please make sure you run this notebook from an Apache Spark 2.3 notebook.
# 
# So the first thing we need to ensure is that we are on the latest version of SystemML, which is 1.3.0 (as of 20th March'19) Please use the code block below to check if you are already on 1.3.0 or higher. 1.3 contains a necessary fix, that's we are running against the SNAPSHOT
# 

# In[1]:


from systemml import MLContext
ml = MLContext(spark)
ml.version()


# 
# 
# If you are blow version 1.3.0, or you got the error message "No module named 'systemml'"  please execute the next two code blocks and then
# 
# # PLEASE RESTART THE KERNEL !!!
# 
# Otherwise your changes won't take effect, just double-check every time you run this notebook if you are on SystemML 1.3
# 

# In[1]:


get_ipython().system('pip install https://github.com/IBM/coursera/blob/master/systemml-1.3.0-SNAPSHOT-python.tar.gz?raw=true')


# 
# 
# Now we need to create two sym links that the newest version is picket up - this is a workaround and will be removed as soon as SystemML 1.3 will be pre-installed on Watson Studio once officially released.
# 

# In[2]:


get_ipython().system('ln -s -f ~/user-libs/python3/systemml/systemml-java/systemml-1.3.0-SNAPSHOT-extra.jar ~/user-libs/spark2/systemml-1.3.0-SNAPSHOT-extra.jar')
get_ipython().system('ln -s -f ~/user-libs/python3/systemml/systemml-java/systemml-1.3.0-SNAPSHOT.jar ~/user-libs/spark2/systemml-1.3.0-SNAPSHOT.jar')


# # Please now restart the kernel and start from the beginning to make sure you've installed SystemML 1.3
# 
# Let's download the test data since it's so small we don't use COS (IBM Cloud Object Store) here

# In[2]:


get_ipython().system('wget https://github.com/IBM/coursera/blob/master/coursera_ml/shake.parquet?raw=true')
get_ipython().system('mv shake.parquet?raw=true shake.parquet')


# Now it’s time to read the sensor data and create a temporary query table.

# In[3]:


df=spark.read.parquet('shake.parquet')


# In[4]:


df.show()


# In[5]:


get_ipython().system('pip install pixiedust')


# In[6]:


import pixiedust
display(df)


# In[7]:


df.createOrReplaceTempView("df")


# We’ll use Apache SystemML to implement Discrete Fourier Transformation. This way all computation continues to happen on the Apache Spark cluster for advanced scalability and performance.

# In[8]:


from systemml import MLContext, dml
ml = MLContext(spark)


# As you’ve learned from the lecture, implementing Discrete Fourier Transformation in a linear algebra programming language is simple. Apache SystemML DML is such a language and as you can see the implementation is straightforward and doesn’t differ too much from the mathematical definition (Just note that the sum operator has been swapped with a vector dot product using the %*% syntax borrowed from R
# ):
# 
# <img style="float: left;" src="https://wikimedia.org/api/rest_v1/media/math/render/svg/1af0a78dc50bbf118ab6bd4c4dcc3c4ff8502223">
# 
# 

# In[9]:


dml_script = '''
PI = 3.141592654
N = nrow(signal)

n = seq(0, N-1, 1)
k = seq(0, N-1, 1)

M = (n %*% t(k))*(2*PI/N)

Xa = cos(M) %*% signal
Xb = sin(M) %*% signal

DFT = cbind(Xa, Xb)
'''


# Now it’s time to create a function which takes a single row Apache Spark data frame as argument (the one containing the accelerometer measurement time series for one axis) and returns the Fourier transformation of it. In addition, we are adding an index column for later joining all axis together and renaming the columns to appropriate names. The result of this function is an Apache Spark DataFrame containing the Fourier Transformation of its input in two columns. 
# 

# In[10]:


from pyspark.sql.functions import monotonically_increasing_id

def dft_systemml(signal,name):
    prog = dml(dml_script).input('signal', signal).output('DFT')
    
    return (

    #execute the script inside the SystemML engine running on top of Apache Spark
    ml.execute(prog) 
     
         #read result from SystemML execution back as SystemML Matrix
        .get('DFT') 
     
         #convert SystemML Matrix to ApacheSpark DataFrame 
        .toDF() 
     
         #rename default column names
        .selectExpr('C1 as %sa' % (name), 'C2 as %sb' % (name)) 
     
         #add unique ID per row for later joining
        .withColumn("id", monotonically_increasing_id())
    )
        



# Now it’s time to create DataFrames containing for each accelerometer sensor axis and one for each class. This means you’ll get 6 DataFrames. Please implement this using the relational API of DataFrames or SparkSQL. Please use class 1 and 2 and not 0 and 1.
# 

# In[11]:


x0 = spark.sql("select X from df where Class=1") #=> Please create a DataFrame containing only measurements of class 0 from the x axis
y0 = spark.sql("select Y from df where Class=1") #=> Please create a DataFrame containing only measurements of class 0 from the y axis
z0 = spark.sql("select Z from df where Class=1") #=> Please create a DataFrame containing only measurements of class 0 from the z axis
x1 = spark.sql("select X from df where Class=2") #=> Please create a DataFrame containing only measurements of class 1 from the x axis
y1 = spark.sql("select Y from df where Class=2") #=> Please create a DataFrame containing only measurements of class 1 from the y axis
z1 = spark.sql("select Z from df where Class=2") #=> Please create a DataFrame containing only measurements of class 1 from the z axis


# Since we’ve created this cool DFT function before, we can just call it for each of the 6 DataFrames now. And since the result of this function call is a DataFrame again we can use the pyspark best practice in simply calling methods on it sequentially. So what we are doing is the following:
# 
# - Calling DFT for each class and accelerometer sensor axis.
# - Joining them together on the ID column. 
# - Re-adding a column containing the class index.
# - Stacking both Dataframes for each classes together
# 
# 

# In[12]:


from pyspark.sql.functions import lit

df_class_0 = dft_systemml(x0,'x')     .join(dft_systemml(y0,'y'), on=['id'], how='inner')     .join(dft_systemml(z0,'z'), on=['id'], how='inner')     .withColumn('class', lit(0))
    
df_class_1 = dft_systemml(x1,'x')     .join(dft_systemml(y1,'y'), on=['id'], how='inner')     .join(dft_systemml(z1,'z'), on=['id'], how='inner')     .withColumn('class', lit(1))

df_dft = df_class_0.union(df_class_1)

df_dft.show()


# Please create a VectorAssembler which consumes the newly created DFT columns and produces a column “features”
# 

# In[13]:


from pyspark.ml.feature import VectorAssembler


# In[14]:


vectorAssembler = VectorAssembler(inputCols=["xa", "xb", "ya", "yb", "za", "zb"], outputCol="features")


# Please insatiate a classifier from the SparkML package and assign it to the classifier variable. Make sure to set the “class” column as target.
# 

# In[15]:


from pyspark.ml.classification import RandomForestClassifier


# In[16]:


classifier = RandomForestClassifier(labelCol="class", featuresCol="features", numTrees=10)


# Let’s train and evaluate…
# 

# In[17]:


from pyspark.ml import Pipeline
pipeline = Pipeline(stages=[vectorAssembler, classifier])


# In[18]:


model = pipeline.fit(df_dft)


# In[19]:


prediction = model.transform(df_dft)


# In[20]:


prediction.show()


# In[21]:


from pyspark.ml.evaluation import MulticlassClassificationEvaluator
binEval = MulticlassClassificationEvaluator().setMetricName("accuracy") .setPredictionCol("prediction").setLabelCol("class")
    
binEval.evaluate(prediction) 


# If you are happy with the result (I’m happy with > 0.8) please submit your solution to the grader by executing the following cells, please don’t forget to obtain an assignment submission token (secret) from the Courera’s graders web page and paste it to the “secret” variable below, including your email address you’ve used for Coursera. 
# 

# In[22]:


get_ipython().system('rm -Rf a2_m4.json')


# In[23]:


prediction = prediction.repartition(1)
prediction.write.json('a2_m4.json')


# In[24]:


get_ipython().system('rm -f rklib.py')
get_ipython().system('wget wget https://raw.githubusercontent.com/IBM/coursera/master/rklib.py')


# In[25]:


from rklib import zipit
zipit('a2_m4.json.zip','a2_m4.json')


# In[26]:


get_ipython().system('base64 a2_m4.json.zip > a2_m4.json.zip.base64')


# In[28]:


from rklib import submit
key = "-fBiYHYDEeiR4QqiFhAvkA"
part = "IjtJk"
email = "dxljack@gmail.com"
submission_token = "IkYeB4tVD3IgGs9f"

with open('a2_m4.json.zip.base64', 'r') as myfile:
    data=myfile.read()
submit(email, submission_token, key, part, [part], data)


# In[ ]:




