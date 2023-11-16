# -*- coding: utf-8 -*-
"""Sprint2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1qUvBXKz4-ZjBUQ3-jyvcMqLGIaLkrdlS
"""



from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

"""Training the model using Random Forest"""

from sklearn.ensemble import RandomForestClassifier
rf_model=RandomForestClassifier()
rf_model.fit(x_train,y_train)

"""**Model Evaluation**

Accuracy of Training Data
"""

x_train_prediction_rf=rf_model.predict(x_train)
train_data_accuracy_rf=accuracy_score(x_train_prediction_rf,y_train)

print('Accuracy: ',train_data_accuracy_rf)

"""Accuracy of Testing Data"""

x_test_prediction_rf=rf_model.predict(x_test)
test_data_accuracy_rf=accuracy_score(x_test_prediction_rf,y_test)

print('Accuracy: ',test_data_accuracy_rf)

"""Training the Model Using SVM"""

from sklearn import svm
classifier=svm.SVC(kernel='linear')

classifier.fit(x_train,y_train)

"""**Model Evaluation**

Accuracy of Training Data
"""

x_train_prediction=classifier.predict(x_train)
training_data_accuracy=accuracy_score(x_train_prediction,y_train)

print('Accuracy: ',training_data_accuracy)

"""Accuracy of Testing Data"""

x_test_prediction=classifier.predict(x_test)
test_data_accuracy=accuracy_score(x_test_prediction,y_test)

print('Accuracy: ',test_data_accuracy)

from sklearn.metrics import classification_report

"""Classification Report Random Forest"""

report = classification_report(y_test, x_test_prediction_rf)
print(report)

"""Classification Report SVM"""

report = classification_report(y_test, x_test_prediction)
print(report)

encoded_region=encoder2.transform(['Madhya Pradesh'])[0]
print(encoded_region)

encoded_location_name=encoder1.transform(['Ashoknagar'])[0]
print(encoded_location_name)

new_data = pd.DataFrame({'location_name':[encoded_region],'region':[encoded_location_name],'latitude':[24.57],'longitude':[77.8],'temperature_celsius': [27.0],'wind_kph':[15.5], 'pressure_mb': [1008],'precip_mm':[0],'humidity':[70],'cloud':[19],'visibility_km':[10]})
predicted_weather = rf_model.predict(new_data)
print(f'Predicted Weather Condition: {q}')