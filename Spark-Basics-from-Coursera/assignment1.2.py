#!/usr/bin/env python
# coding: utf-8

# # Welcome
# Welcome the the 1st course on coursera called "A developer's guide to Exploratory Analysis of IoT Sensor Data" which can be found here: https://www.coursera.org/teach/developer-iot-data-analyst-exploratory
# 

# The intention of this template is to give a framework where the individual programming assignments can be implemented. This is the one for Week 1

# # Question 1
# Below you see some ApacheSpark code written in Python which will be picked up by the auto grader of coursera.org. You don't have to change code now, the only thing we want you to do is export this notebook as python code so that the grader can assess it. This is an exercice ment to make sure the submission process works on your side.
# 
# PLEASE DON'T ADD ANY CODE OUTSIDE THE assignment1 FUNCTION

# In[1]:


def assignment1(sc):
    rdd = sc.parallelize(range(100))
    return rdd.count()


# In[11]:


### PLEASE DON'T REMOVE THIS BLOCK - THE FOLLOWING CODE IS NOT GRADED
#axx
### PLEASE DON'T REMOVE THIS BLOCK - THE FOLLOWING CODE IS NOT GRADED


# In[3]:


print(assignment1(sc))

