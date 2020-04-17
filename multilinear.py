# Import libraries.
import numpy as np
from sklearn.linear_model import LinearRegression

# Defining input and output datapoints.
x = [[1, 5], [0, 3], [11, 9], [15, 5], [35, 15], [25, 15], [55, 45], [60, 20]]
y = [4, 6, 24, 8, 32, 28, 42, 47]
x, y = np.array(x), np.array(y)


# Displaying NumPy array.
print("The values stored in x variable are:")
print(x)
print("\n")
print("The values stored in y variable are:")
print(y)

#Creating a multi-linear regression model.
model = LinearRegression().fit(x, y)

# Computing result. 
r_sq = model.score(x, y)
print('coefficient of determination:', r_sq)
print('\n')
print('intercept:', model.intercept_)
print('\n')
print('slope:', model.coef_)

#Making prediction.
y_pred = model.predict(x) 
print('predicted response:', y_pred, sep='\n')


#Evaluating model.
x_new = np.arange(10).reshape((-1, 2))
print("New input x values")
print(x_new)
print("\n")
print("New predicted y values")
y_new = model.predict(x_new)
print(y_new)


