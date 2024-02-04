import pickle
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from skimage.io import imread
from skimage.transform import resize
from sklearn.metrics import accuracy_score

input_dir = "/Users/antonytce/Desktop/CV/clf-data"
categories = ["empty", "not_empty"]

data = []
labels = []

for category_idx, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir, category)):
        img_path = os.path.join(input_dir, category, file)

        img = imread(img_path)
        img = resize(img, (15, 15))

        data.append(img.flatten())
        labels.append(category_idx)

data = np.asarray(data)
labels = np.asarray(labels)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

csv_classifier = SVC()
parameters = [{"gamma": [0.01, 0.001, 0.0001], "C": [1, 10, 100, 1000]}]

grid_search = GridSearchCV(csv_classifier, parameters)
grid_search.fit(X_train, y_train)

best_est = grid_search.best_estimator_
y_pred = best_est.predict(X_test)

print(accuracy_score(y_test, y_pred))

pickle.dump(best_est, open("./model.p", "wb"))