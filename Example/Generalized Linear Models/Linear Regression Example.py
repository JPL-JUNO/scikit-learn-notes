import matplotlib.pylab as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

diabetes_X = diabetes_X[:, np.newaxis, 2]
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

reg = linear_model.LinearRegression()
reg.fit(diabetes_X_train, diabetes_y_train)
diabetes_y_pred = reg.predict(diabetes_X_test)

print('Coefficients: \n', reg.coef_)
print('Mean squared error: {:.2f}'.format(mean_squared_error(diabetes_y_test, diabetes_y_pred)))
print('Coefficient of determination: {:.2f}'.format(r2_score(diabetes_y_test, diabetes_y_pred)))

plt.scatter(diabetes_X_test, diabetes_y_test, color='black')
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()
