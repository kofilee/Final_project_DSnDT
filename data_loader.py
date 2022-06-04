import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split


def data_loader(test_size=0.15):
    label_encoder = LabelEncoder()
    one_hot_encoder = OneHotEncoder()
    scaler = StandardScaler()

    df = pd.read_excel("Date_Fruit_Datasets.xlsx")
    X = df.iloc[:, 0:-2]
    Y = df.iloc[:, -1]
    Y_label = label_encoder.fit_transform(Y)
    Y_one_hot = one_hot_encoder.fit_transform(Y_label.reshape(-1, 1))
    x = X.values
    x_scaled = scaler.fit_transform(x)
    X_train, X_test, Y_train, Y_test = train_test_split(x_scaled, Y_one_hot.toarray(), test_size=test_size,
                                                        random_state=72)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.10,
                                                      random_state=72)
    label = label_encoder.inverse_transform(range(7))
    return X_train, X_val, X_test, Y_train, Y_val, Y_test, label
