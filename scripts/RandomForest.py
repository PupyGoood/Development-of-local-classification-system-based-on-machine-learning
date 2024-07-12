import warnings

# ignore all future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


def run(file_name, testing_percentage, n_estimators, criterion, max_features, feature_scaling, apply_pca,
        pca_components):
    df = pd.read_csv(file_name)
    ts = (testing_percentage) / 100

    x = df.drop(columns=['Class'])
    y = pd.DataFrame(df['Class'])

    X = x.iloc[:, :].values
    Y = y.iloc[:, :].values.ravel()  # Use ravel() to convert to 1D array

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=ts, random_state=109)

    if feature_scaling == "StandardScaler":
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
    elif feature_scaling == "MinMaxScaler":
        sc = MinMaxScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

    if apply_pca:
        pca_components = min(pca_components, X_train.shape[1])
        pca = PCA(n_components=pca_components)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    clf = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_features=max_features)
    clf.fit(X_train, Y_train)

    Y_predict = clf.predict(X_test)

    # Get unique classes in the dataset
    unique_classes = np.unique(np.concatenate((Y_train, Y_test)))
    cm = confusion_matrix(Y_test, Y_predict, labels=unique_classes)

    confusion = pd.DataFrame(cm, index=[f'actual {cls}' for cls in unique_classes],
                             columns=[f'predicted {cls}' for cls in unique_classes])

    sns.heatmap(confusion, annot=True, fmt='d')
    plt.show()

    results = classification_report(Y_test, Y_predict, target_names=[str(cls) for cls in unique_classes])

    return results

# Example usage:
# print(run('C:/Users/user/Documents/pythonprog/ML/MLGUI/scripts/bill_authentication.csv', 20, 100, 'gini', 'auto', 'StandardScaler', True, 2))
