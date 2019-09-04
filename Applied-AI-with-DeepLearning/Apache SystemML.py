#!/usr/bin/env python
# coding: utf-8

# # Assignment 4
# ## Understaning scaling of linear algebra operations on Apache Spark using Apache SystemML
# 
# In this assignment we want you to understand how to scale linear algebra operations from a single machine to multiple machines, memory and CPU cores using Apache SystemML. Therefore we want you to understand how to migrate from a numpy program to a SystemML DML program. Don't worry. We will give you a lot of hints. Finally, you won't need this knowledge anyways if you are sticking to Keras only, but once you go beyond that point you'll be happy to see what's going on behind the scenes. Please make sure you run this notebook from an Apache Spark 2.3 notebook.
# 
# So the first thing we need to ensure is that we are on the latest version of SystemML, which is 1.2.0 (as of Feb. '19)
# Please use the code block below to check if you are already on 1.2.0 or higher.

# In[2]:


from systemml import MLContext
ml = MLContext(spark)
ml.version()


# If you are blow version 1.2.0 please execute the next two code blocks

# In[2]:


get_ipython().system('pip install systemml')


# Now we need to create two sym links that the newest version is picket up - this is a workaround and will be removed soon

# In[4]:


get_ipython().system('ln -s -f ~/user-libs/python3/systemml/systemml-java/systemml-1.2.0-extra.jar ~/user-libs/spark2/systemml-1.2.0-extra.jar')
get_ipython().system('ln -s -f ~/user-libs/python3/systemml/systemml-java/systemml-1.2.0.jar ~/user-libs/spark2/systemml-1.2.0.jar')


# Now please restart the kernel and make sure the version is correct

# In[3]:


from systemml import MLContext
ml = MLContext(spark)
ml.version()


# Congratulations, if you see version 1.2.0 or higher, please continue with the notebook...

# In[3]:


from systemml import MLContext, dml
import numpy as np
import time


# Then we create an MLContext to interface with Apache SystemML. Note that we pass a SparkSession object as parameter so SystemML now knows how to talk to the Apache Spark cluster

# In[4]:


ml = MLContext(spark)


# Now we create some large random matrices to have numpy and SystemML crunch on it

# In[5]:


u = np.random.rand(1000,10000)
s = np.random.rand(10000,1000)
w = np.random.rand(1000,1000)


# Now we implement a short one-liner to define a very simple linear algebra operation
# 
# In case you are unfamiliar with matrxi-matrix multiplication: https://en.wikipedia.org/wiki/Matrix_multiplication
# 
# sum(U' * (W . (U * S)))
# 
# 
# | Legend        |            |   
# | ------------- |-------------| 
# | '      | transpose of a matrix | 
# | * | matrix-matrix multiplication      |  
# | . | scalar multiplication      |   
# 
# 

# In[6]:


start = time.time()
res = np.sum(u.T.dot(w * u.dot(s)))
print (time.time()-start)


# As you can see this executes perfectly fine. Note that this is even a very efficient execution because numpy uses a C/C++ backend which is known for it's performance. But what happens if U, S or W get such big that the available main memory cannot cope with it? Let's give it a try:

# In[ ]:


#u = np.random.rand(10000,100000)
#s = np.random.rand(100000,10000)
#w = np.random.rand(10000,10000)


# After a short while you should see a memory error. This is because the operating system process was not able to allocate enough memory for storing the numpy array on the heap. Now it's time to re-implement the very same operations as DML in SystemML, and this is your task. Just replace all ###your_code_goes_here sections with proper code, please consider the following table which contains all DML syntax you need:
# 
# | Syntax        |            |   
# | ------------- |-------------| 
# | t(M)      | transpose of a matrix, where M is the matrix | 
# | %*% | matrix-matrix multiplication      |  
# | * | scalar multiplication      |   
# 
# ## Task

# In order to show you the advantage of SystemML over numpy we've blown up the sizes of the matrices. Unfortunately, on a 1-2 worker Spark cluster it takes quite some time to complete. Therefore we've stripped down the example to smaller matrices below, but we've kept the code, just in case you are curious to check it out. But you might want to use some more workers which you easily can configure in the environment settings of the project within Watson Studio. Just be aware that you're currently limited to free 50 capacity unit hours per month wich are consumed by the additional workers.

# In[10]:


script = """
U = rand(rows=1000,cols=10000, seed=5)
S = rand(rows=10000,cols=1000, seed=23)
W = rand(rows=1000,cols=1000, seed=42)
res = sum(t(U) %*% (W * (U %*% S)))
"""


# To get consistent results we switch from a random matrix initialization to something deterministic

# In[11]:


prog = dml(script).output('res')
res = ml.execute(prog).get('res')
print(res)


# If everything runs fine you should get *6252492444241.075* as result (or something in that bullpark). Feel free to submit your DML script to the grader now!
# 
# ### Submission

# In[12]:


get_ipython().system('rm -f rklib.py')
get_ipython().system('wget https://raw.githubusercontent.com/romeokienzler/developerWorks/master/coursera/ai/rklib.py')


# In[14]:


from rklib import submit
key = "esRk7vn-Eeej-BLTuYzd0g"


email = "dxljack@gmail.com"


# In[15]:


part = "fUxc8"
secret = "FiNLW2wd4atQxx2F"
submit(email, secret, key, part, [part], script)


# In[ ]:




