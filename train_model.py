import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

all_d = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(all_d['data'])
labels = np.asarray(all_d['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = SVC(C=2, kernel='rbf')

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

score = accuracy_score(y_pred, y_test)

print(f"accuracy: {score}")

file = open("./model.pickle", 'wb')
pickle.dump({'model': model}, file)
file.close()
