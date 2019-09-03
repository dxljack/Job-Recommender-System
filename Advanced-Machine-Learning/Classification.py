#!/usr/bin/env python
# coding: utf-8

# This is the second assignment for the Coursera course "Advanced Machine Learning and Signal Processing"
# 
# 
# Just execute all cells one after the other and you are done - just note that in the last one you have to update your email address (the one you've used for coursera) and obtain a submission token, you get this from the programming assignment directly on coursera.
# 
# Please fill in the sections labelled with "###YOUR_CODE_GOES_HERE###"

# In[24]:


get_ipython().system('wget https://github.com/IBM/coursera/raw/master/coursera_ml/a2.parquet')


# Now it’s time to have a look at the recorded sensor data. You should see data similar to the one exemplified below….
# 

# In[36]:


df=spark.read.load('a2.parquet')

df.createOrReplaceTempView("df")
spark.sql("SELECT * from df").show()


# Please create a VectorAssembler which consumes columns X, Y and Z and produces a column “features”
# 

# In[37]:


from pyspark.ml.feature import VectorAssembler
vectorAssembler = VectorAssembler(inputCols=["X", "Y", "Z"], outputCol="features")


# Please instantiate a classifier from the SparkML package and assign it to the classifier variable. Make sure to either
# 1.	Rename the “CLASS” column to “label” or
# 2.	Specify the label-column correctly to be “CLASS”
# 

# In[42]:


from pyspark.ml.classification import GBTClassifier

classifier = GBTClassifier(featuresCol="features", labelCol="CLASS")


# Let’s train and evaluate…
# 

# In[43]:


from pyspark.ml import Pipeline
pipeline = Pipeline(stages=[vectorAssembler, classifier])


# In[44]:


model = pipeline.fit(df)


# In[45]:


prediction = model.transform(df)


# In[46]:


prediction.show()


# In[47]:


from pyspark.ml.evaluation import MulticlassClassificationEvaluator
binEval = MulticlassClassificationEvaluator().setMetricName("accuracy") .setPredictionCol("prediction").setLabelCol("CLASS")
    
binEval.evaluate(prediction) 


# If you are happy with the result (I’m happy with > 0.55) please submit your solution to the grader by executing the following cells, please don’t forget to obtain an assignment submission token (secret) from the Coursera’s graders web page and paste it to the “secret” variable below, including your email address you’ve used for Coursera. (0.55 means that you are performing better than random guesses)
# 

# In[48]:


get_ipython().system('rm -Rf a2_m2.json')


# In[49]:


prediction = prediction.repartition(1)
prediction.write.json('a2_m2.json')


# In[50]:


get_ipython().system('rm -f rklib.py')
get_ipython().system('wget https://raw.githubusercontent.com/IBM/coursera/master/rklib.py')


# In[51]:


import zipfile

def zipdir(path, ziph):
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))

zipf = zipfile.ZipFile('a2_m2.json.zip', 'w', zipfile.ZIP_DEFLATED)
zipdir('a2_m2.json', zipf)
zipf.close()


# In[52]:


get_ipython().system('base64 a2_m2.json.zip > a2_m2.json.zip.base64')


# In[53]:


from rklib import submit
key = "J3sDL2J8EeiaXhILFWw2-g"
part = "G4P6f"
email = "dxljack@gmail.com"
secret = "pxeYG07xFWj2exAO"

with open('a2_m2.json.zip.base64', 'r') as myfile:
    data=myfile.read()
submit(email, secret, key, part, [part], data)


# In[ ]:




