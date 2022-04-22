import os
import pandas as pd
import aifeynman
import tempfile, os, pdb, csv, traceback,random, time
import numpy as np
from func_timeout import func_timeout, FunctionTimedOut

class Problem:
    def __init__(self, row):
        self.eq_id      = row['Filename']
        self.form       = row['Formula']
        self.n_vars     = int(row['# variables'])
        self.var_names  = [row[f'v{i+1}_name']  for i in range(self.n_vars)] 
        self.low        = [float(row[f'v{i+1}_low'])   for i in range(self.n_vars)]
        self.high       = [float(row[f'v{i+1}_high'])  for i in range(self.n_vars)]
        self.dp         = int(row[f'datapoints'])
        
        self.X = []
        self.Y = []
        for i in range(self.dp):
            x = []
            d = {}
            for _ in range(len(self.var_names)):
                v = np.random.uniform(self.low[_], self.high[_])
                x.append(v)
                d[self.var_names[_]] = np.random.uniform(self.low[_], self.high[_])
            self.X.append(x)
            d['exp'] = np.exp
            d['sqrt'] = np.sqrt
            d['pi'] = np.pi
            d['cos'] = np.cos
            d['sin'] = np.sin
            d['tan'] = np.tan
            d['tanh'] = np.tanh
            d['ln']   = np.log
            d['arcsin'] = np.arcsin
            self.Y.append(eval(self.form,d))

# def mk_problems():
#     ret = [] 
#     with open('old.csv') as csvfile:
#         reader = csv.DictReader(csvfile)
#         for row in reader:
#             if row['Filename'] == '': continue
#             try:
#                 p = Problem(row)
#                 ret.append(p)
#             except Exception as e:
#                 traceback.print_exc()
#                 print(row)
#                 print("FAIL")

def make_dataset(X, Y, path, eq_name):
    assert len(X) == len(Y)

    with open(path + eq_name, 'w') as tf:
        for it in range(len(X)):
            for val in X[it]:
                tf.write(f"{val}\t")
            tf.write(f"{Y[it]}\n")
        tf.close()
        tf_directory = os.path.dirname(tf.name)


def feynman_solve(X,Y):
    assert len(X) == len(Y), 'uneven X,Y'
    with tempfile.NamedTemporaryFile(suffix='.csv', prefix=os.path.basename(__file__), mode = "w", delete=False) as tf:
        for it in range(len(X)):
            for val in X[it]:
                tf.write(f"{val}\t")
            tf.write(f"{Y[it]}\n")
        tf.close()
        tf_directory = os.path.dirname(tf.name)
    # return run_aifeynman('', tf.name, BF_try_time=30,BF_ops_file_type='19ops.txt', polyfit_deg=4, NN_epochs=4000)[-1][-1]

if __name__ == '__main__':

    dir_path = 'data/'

    x = pd.read_csv('old.csv')
    df = pd.DataFrame(x)

    row = df.iloc[7]


    p = Problem(row)
    X = p.X
    Y = p.Y
    eq_id = p.eq_id
    
    # make_dataset(X, Y, dir_path, eq_id)

    aifeynman.run_aifeynman('', 'data/I.12.1', 60, "14ops.txt", polyfit_deg=3, NN_epochs=500)


    # feynman_solve(X, Y)