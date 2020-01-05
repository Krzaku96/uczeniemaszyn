import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from common.import_data import ImportData
from sklearn.model_selection import KFold

if __name__ == "__main__":
    data_set = ImportData()

    kf = KFold(n_splits=10, shuffle=True)
    #x = data_set.import_columns(['s1','r1','s2','r2','s3','r3','s4','r4','s5','r5'])
    x: np.ndarray = data_set.import_columns(['V1', 'V2', 'V3','V4'])
    #x = data_set.import_all_data()
    y = data_set.import_columns(np.array(['Class']))
    print(x)


    scores = []
    for i in range(5):
        result = next(kf.split(x), None)
        x_train = x[result[0]]
        x_test = x[result[1]]
        y_train = y[result[0]]
        y_test = y[result[1]]
        NN = KNeighborsClassifier(n_neighbors=5)
        NN.fit(x_train,y_train.ravel())
        predictions = NN.predict(x_test)
        scores.append(NN.score(x_test, y_test))
        print('Scores from each Iteration: ', scores)

    print('Average K-Fold Score :', np.mean(scores))