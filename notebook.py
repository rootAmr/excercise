#!/usr/bin/env python
# coding: utf-8

# In[1]:


"Hello My Name is Maulana"


# In[9]:


# ## Data Science Languages

# 1. Python
# 2. R
# 3. Julia
# 4. SQL
# 5. Scala
# 6. MATLAB
# 7. Java
# 8. C/C++
# 9. SAS
# 10. JavaScript


# In[10]:


## Data Science Libraries

# - NumPy
# - Pandas
# - Matplotlib
# - Seaborn
# - Scikit-learn
# - TensorFlow
# - Keras
# - PyTorch
# - NLTK (Natural Language Toolkit)
# - SciPy
# - Statsmodels
# - XGBoost
# - LightGBM


# In[8]:


import pandas as pd

# Create a sample DataFrame
data = {
    'Name': ['John', 'Alice', 'Bob', 'Eva', 'Mark'],
    'Age': [25, 30, 22, 28, 35],
    'City': ['New York', 'San Francisco', 'Chicago', 'Los Angeles', 'Seattle']
}

df = pd.DataFrame(data)

# Display the DataFrame
print("Pandas DataFrame:")
print(df)

# Perform operations on the DataFrame
average_age = df['Age'].mean()
youngest_person = df.loc[df['Age'].idxmin()]

print("Average Age:", average_age)
print("Youngest Person:")
print(youngest_person)


# In[ ]:


import matplotlib.pyplot as plt

# Sample data for plotting
x = [1, 2, 3, 4, 5]
y = [10, 12, 8, 15, 9]

# Line plot
plt.plot(x, y)
plt.title('Line Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()

# Bar plot
plt.bar(x, y)
plt.title('Bar Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()


# In[ ]:


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a Logistic Regression model
model = LogisticRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)


# In[ ]:





# In[6]:


from tabulate import tabulate

# Data for the table
data = [
    ["Tool", "Description"],
    ["Python", "Versatile programming language for data analysis, machine learning, and visualization."],
    ["R", "Powerful statistical programming language for data manipulation and analysis."],
    ["SQL", "Language for managing and querying relational databases."],
    ["Jupyter Notebook", "Interactive computing environment for code, visualizations, and text."],
    ["Pandas", "Python library for data manipulation and analysis."],
    ["NumPy", "Fundamental package for numerical computing in Python."],
]

# Convert data to markdown table format
table = tabulate(data, tablefmt="pipe")

# Print the markdown table
print(table)


# In[7]:


# Addition
result_addition = 2 + 3
print("Addition Result:", result_addition)

# Subtraction
result_subtraction = 10 - 5
print("Subtraction Result:", result_subtraction)

# Multiplication
result_multiplication = 4 * 6
print("Multiplication Result:", result_multiplication)

# Division
result_division = 15 / 3
print("Division Result:", result_division)

# Exponentiation
result_exponentiation = 2 ** 3
print("Exponentiation Result:", result_exponentiation)

# Modulo
result_modulo = 17 % 5
print("Modulo Result:", result_modulo)

# Complex Expression
complex_expression_result = (3 + 5) * 2 - (10 / 2)
print("Complex Expression Result:", complex_expression_result)


# In[11]:


# Multiplication
num1 = 5
num2 = 10
result_multiplication = num1 * num2
print("Multiplication Result:", result_multiplication)

# Addition
num3 = 15
num4 = 20
result_addition = num3 + num4
print("Addition Result:", result_addition)


# In[12]:


# Function to convert minutes to hours
def convert_minutes_to_hours(minutes):
    hours = minutes / 60
    return hours

# Example usage
minutes = 150
hours = convert_minutes_to_hours(minutes)
print(f"{minutes} minutes is equal to {hours:.2f} hours")


# In[ ]:


# ## Objectives

# The main objectives of this notebook are:

# 1. Introduce data science libraries and tools commonly used in the field.
# 2. Provide examples of basic arithmetic operations in Python.
# 3. Demonstrate the usage of data manipulation and analysis with Pandas.
# 4. Showcase data visualization using Matplotlib and Seaborn.
# 5. Illustrate machine learning with Scikit-learn and TensorFlow.
# 6. Explore deep learning concepts using Keras and PyTorch.
# 7. Introduce natural language processing with NLTK.
# 8. Showcase statistical analysis with SciPy and Statsmodels.
# 9. Demonstrate boosting algorithms with XGBoost and LightGBM.
# 10. Present practical data science scenarios and use cases.

# Throughout this notebook, we will work on various code examples and explanations to achieve these objectives. Let's get started!


# In[13]:


## Author

Ahmad Maulana Rismadin


# In[ ]:




