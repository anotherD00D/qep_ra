import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from matplotlib import pyplot as plt
import matlab.engine


iris = load_iris()
eng = matlab.engine.start_matlab()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

df['flower_name'] = df.target.apply(lambda x: iris.target_names[x])

df0 = df[df.target == 0]
df1 = df[df.target == 1]
df2 = df[df.target == 2]

plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
plt.scatter(df0['petal length (cm)'], df0['petal width (cm)'], color ='green', marker='+')
plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'], color ='blue', marker='+')

X = df[['petal length (cm)', 'petal width (cm)']]
y = df.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = SVC(kernel='linear')
model.fit(X_train, y_train)

w = model.coef_[0]
b = model.intercept_[0]
x_vals = np.linspace(X['petal length (cm)'].min(), X['petal length (cm)'].max(), 200)
y_boundary = -(w[0] / w[1]) * x_vals - b / w[1]

margin = 1 / np.linalg.norm(w)
y_margin_up = -(w[0] / w[1]) * x_vals - (b - 1) / w[1]
y_margin_down = -(w[0] / w[1]) * x_vals - (b + 1) / w[1]

plt.plot(x_vals, y_boundary, 'k-', linewidth=2, label='Decision Boundary')
plt.plot(x_vals, y_margin_up, 'r--', linewidth=1)
plt.plot(x_vals, y_margin_down, 'b--', linewidth=1)

plt.xlim(X['petal length (cm)'].min() - 0.25, X['petal length (cm)'].max() - 1.5)
plt.ylim(X['petal width (cm)'].min() - 0.25, X['petal width (cm)'].max() - 0.5)

print(b)
print(model.score(X_test, y_test))

plt.show()
eng.quit()

