# Liniar regression



#Import Library
#import numpy
#Import other necessary libraries like pandas, numpy...
from sklearn import linear_model

#Load Train and Test datasets
from sklearn import datasets
iris = datasets.load_boston
print(iris)

# one way to split data into Train and Test
import pandas as pd
import numpy as np
df = pd.DataFrame(np.random.randn(100, 2))
msk = np.random.rand(len(df)) < 0.8

x_train = df[msk]
test = df[~msk]
print( len(x_train))
print( len(test))

#another way to split the data in to test and Train
train=df.sample(frac=0.8,random_state=200)
test=df.drop(train.index)
print( len(train))
print( len(test))


#Identify feature and response variable(s) and values must be numeric and numpy arrays
x_train=input_variables_values_training_datasets
y_train=target_variables_values_training_datasets
x_test=input_variables_values_test_datasets

# Create linear regression object
linear = linear_model.LinearRegression()
# Train the model using the training sets and check score
linear.fit(x_train, y_train)
linear.score(x_train, y_train)
#Equation coefficient and Intercept
print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)
#Predict Output
predicted= linear.predict(x_test)


