from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pandas as pd
from matplotlib import pyplot as plt
import matlab.engine

digits = load_digits()
# eng = matlab.engine.start_matlab()

df = pd.DataFrame(digits.data, columns=digits.feature_names)
df['target'] = digits.target
df['target_name'] = df.target.apply(lambda x: digits.target_names[x])

df0 = df[df.target_name==0]
df1 = df[df.target_name==1]

# plt.xlabel('pixel_0_0')
# plt.ylabel('pixel_0_1')
# plt.scatter(df0['pixel_0_0'], df0['pixel_0_1'], color ='green', marker='+')
# plt.scatter(df1['pixel_0_0'], df1['pixel_0_1'], color ='blue', marker='+')

# plt.show()
# eng.quit()


X = df.drop(['target', 'target_name'], axis='columns')
y = df.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = SVC()

model.fit(X_train, y_train)

print(model.score(X_test, y_test))