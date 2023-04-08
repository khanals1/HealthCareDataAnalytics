import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def getPrediction(df):
    categorical_columns = ["race", "gender", "age", "admission_type_id", "discharge_disposition_id",
                           "admission_source_id", "time_in_hospital","payer_code","medical_specialty", "num_lab_procedures",
                           "num_procedures", "num_medications", "number_outpatient",
                           "number_emergency", "number_inpatient", "diag_1", "diag_2", "diag_3",
                           "number_diagnoses", #"metformin","glipizide","glyburide","pioglitazone","rosiglitazone", "insulin","change",
                           "diabetesmed", "readmitted"]

    df = pd.get_dummies(df, columns=categorical_columns)

    X = df.drop(['readmitted_Yes', 'readmitted_No'], axis=1)
    y = df['readmitted_Yes']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42, stratify=True)
    decision_tree_classifier = DecisionTreeClassifier()
    decision_tree_classifier.fit(X_train, y_train)
    y_prediction = decision_tree_classifier.predict(X_test)
    results = metrics.confusion_matrix(y_test, y_prediction)
    model_accuracy = metrics.classification_report(y_test, y_prediction)
    print("Predicting Readmission from Decision Tree Classifier: \n")
    print(results)
    print(model_accuracy)

    gaussian_model = GaussianNB()
    gaussian_model.fit(X_train, y_train)
    y_prediction = gaussian_model.predict(X_test)
    results = metrics.confusion_matrix(y_test, y_prediction)
    model_accuracy = metrics.classification_report(y_test, y_prediction)
    print("Predicting Readmission from Naive Bayes Model: \n")
    print(results)
    print(model_accuracy)

    logreg = LogisticRegression(C=1)
    logreg.fit(X_train, y_train)
    y_prediction = logreg.predict(X_test)
    results = metrics.confusion_matrix(y_test, y_prediction)
    model_accuracy = metrics.classification_report(y_test, y_prediction)
    print("Predicting Readmission from from Logistic Regression: \n")
    print(results)
    print(model_accuracy)

    model = Sequential()
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train,
                        validation_data=(X_test, y_test),
                        epochs=50,
                        batch_size=2048,
                        verbose=1)
    print("Predicting Readmission from NN:")
    print(model.evaluate(X_test, y_test)[1])
    _, accuracy = model.evaluate(X_test, y_test)
    print('Accuracy: %.2f' % (accuracy * 100))

    X = df.drop(['diabetesmed_Yes', 'diabetesmed_No', 'readmitted_Yes', 'readmitted_No'], axis=1)
    y = df['diabetesmed_Yes']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    decision_tree_classifier = DecisionTreeClassifier()
    decision_tree_classifier.fit(X_train, y_train)
    y_prediction = decision_tree_classifier.predict(X_test)
    results = metrics.confusion_matrix(y_test, y_prediction)
    model_accuracy = metrics.classification_report(y_test, y_prediction)
    print("Predicting Diabetes from Decision Tree Classifier: \n")
    print(results)
    print(model_accuracy)

    gaussian_model = GaussianNB()
    gaussian_model.fit(X_train, y_train)
    y_prediction = gaussian_model.predict(X_test)
    results = metrics.confusion_matrix(y_test, y_prediction)
    model_accuracy = metrics.classification_report(y_test, y_prediction)
    print("Predicting Diabetes from Naive Bayes Model: \n")
    print(results)
    print(model_accuracy)

    logreg = LogisticRegression(C=1)
    logreg.fit(X_train, y_train)
    y_prediction = logreg.predict(X_test)
    results = metrics.confusion_matrix(y_test, y_prediction)
    model_accuracy = metrics.classification_report(y_test, y_prediction)
    print("Predicting Diabetes from from Logistic Regression: \n")
    print(results)
    print(model_accuracy)

    model = Sequential()
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train,
                        validation_data=(X_test, y_test),
                        epochs=50,
                        batch_size=2048,
                        verbose=1)
    print("Predicting Diabetes from NN:")
    print(model.evaluate(X_test, y_test)[1])
    _, accuracy = model.evaluate(X_test, y_test)
    print('Accuracy: %.2f' % (accuracy * 100))
    return df

"""
    X = df.loc[:, ~df.columns.str.startswith('discharge_disposition_id')]
    y = df['discharge_disposition_id_Expired']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    decision_tree_classifier = DecisionTreeClassifier()
    decision_tree_classifier.fit(X_train, y_train)
    y_prediction = decision_tree_classifier.predict(X_test)
    results = metrics.confusion_matrix(y_test, y_prediction)
    model_accuracy = metrics.classification_report(y_test, y_prediction)
    print("Predicting Death from Decision Tree Classifier: \n")
    print(results)
    print(model_accuracy)

    gaussian_model = GaussianNB()
    gaussian_model.fit(X_train, y_train)
    y_prediction = gaussian_model.predict(X_test)
    results = metrics.confusion_matrix(y_test, y_prediction)
    model_accuracy = metrics.classification_report(y_test, y_prediction)
    print("Predicting Death from Naive Bayes Model: \n")
    print(results)
    print(model_accuracy)

    logreg = LogisticRegression(C=1)
    logreg.fit(X_train, y_train)
    y_prediction = logreg.predict(X_test)
    results = metrics.confusion_matrix(y_test, y_prediction)
    model_accuracy = metrics.classification_report(y_test, y_prediction)
    print("Predicting Death from from Logistic Regression: \n")
    print(results)
    print(model_accuracy)

    model = Sequential()
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(1028, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train,
                        validation_data=(X_test, y_test),
                        epochs=60,
                        batch_size=256,
                        verbose=1)
    print("Predicting Death from NN:")
    print(model.evaluate(X_test, y_test)[1])
    _, accuracy = model.evaluate(X_test, y_test)
    print('Accuracy: %.2f' % (accuracy * 100))
"""
