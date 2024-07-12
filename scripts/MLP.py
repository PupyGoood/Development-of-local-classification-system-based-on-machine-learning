import warnings
# ignore all future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")
#importing math modules
from math import sqrt

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix


# Keras specific
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import seaborn as sns


def make_model(input_dim,output_dim):
    temp_model = Sequential()
    temp_model.add(Dense(50, activation='relu', input_dim=input_dim))
    temp_model.add(Dense(10, activation='relu'))
    temp_model.add(Dense(5, activation='relu'))
    temp_model.add(Dense(output_dim, activation='softmax'))
    return temp_model



def run(file_name,model, testing_percentage=20,
        target_col='Class', optimizer='adam',epochs=10, feature_scaling= None, apply_pca= False, pca_components = 2):
    df = pd.read_csv(file_name)
    df.describe()

    ts = (testing_percentage)/100
    target_column = [target_col]
    predictors = list(set(list(df.columns))-set(target_column))
    df.describe()

    X = df[predictors].values
    y = df[target_column].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=ts, random_state=40)

    if feature_scaling == "StandardScaler":
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        Y_train = to_categorical(Y_train)
        Y_test = to_categorical(Y_test)
    elif feature_scaling == "MinMaxScaler":
        sc = MinMaxScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        Y_train = to_categorical(Y_train)
        Y_test = to_categorical(Y_test)
    if apply_pca:
        pca_components = min(pca_components, X_train.shape[1])
        pca = PCA(n_components=pca_components)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    count_classes = Y_test.shape[1]
    # print(count_classes)

    model = make_model(len(predictors),count_classes)
    # Compile the model
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(X_train, Y_train, epochs=epochs)

    pred_test = model.predict(X_test)

    results = classification_report(Y_test.argmax(axis=1), pred_test.argmax(axis=1))

    unique_classes = np.unique(np.concatenate((Y_train.argmax(axis=1), Y_test.argmax(axis=1))))
    cm = confusion_matrix(Y_test.argmax(axis=1), pred_test.argmax(axis=1), labels=unique_classes)

    confusion = pd.DataFrame(cm, index=[f'actual {cls}' for cls in unique_classes],
                             columns=[f'predicted {cls}' for cls in unique_classes])

    # cm = confusion_matrix(Y_test.argmax(axis=1), pred_test.argmax(axis=1),labels=[0,1])

    # confusion = pd.DataFrame(cm, index = ['actual 0', 'actual 1'], columns = ['predicted 0','predicted 1'])
    sns.heatmap(confusion, annot = True, fmt='d')
    plt.figure()

    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')

    plt.show()

    return results

# run()
