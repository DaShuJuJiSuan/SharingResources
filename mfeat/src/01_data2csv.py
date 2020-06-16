import pandas as pd
import numpy as np
def safe_float(number):
    try:
        return float(number)
    except:
        return np.nan
if __name__ == '__main__':
    data = []
    with open("../data/original_data/mfeat-zer","r") as f:
        for lines in f:
            a = lines.strip().split()
            # a = list(map(safe_float,a))
            data.append(a)
            #print(a)
    # print(data)
    data = pd.DataFrame(data, columns=None,index=None)
    data.to_csv("../data/experiment_data/mfeat_zer.csv", index=None, header=None)