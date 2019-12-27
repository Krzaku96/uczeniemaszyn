import numpy as np

from common.import_data import ImportData
from collections import Counter
from imblearn.under_sampling import AllKNN

if __name__ == "__main__":
    data_set = ImportData()
    x: np.ndarray = data_set.import_all_data()
    y: np.ndarray = data_set.import_columns(np.array(['Class'])).ravel()
    print('Original dataset shape %s' % Counter(y))
    allknn = AllKNN()
    x_res, y_res = allknn.fit_resample(x, y)
    print('Reduced dataset shape %s' % Counter(y_res))
