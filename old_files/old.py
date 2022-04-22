from feynman.S_run_aifeynman import run_aifeynman
# from gplearn.genetic import SymbolicRegressor
# from SymbolicRegression.symbolic_regression.tree_model import Tree_Model
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

def mk_problems():
    ret = [] 
    with open('old.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['Filename'] == '': continue
            try:
                p = Problem(row)
                ret.append(p)
            except Exception as e:
                traceback.print_exc()
                print(row)
                print("FAIL")
    return ret
def feynman_solve(X,Y):
    assert len(X) == len(Y), 'uneven X,Y'
    with tempfile.NamedTemporaryFile(suffix='.csv', prefix=os.path.basename(__file__), mode = "w", delete=False) as tf:
        for it in range(len(X)):
            for val in X[it]:
                tf.write(f"{val}\t")
            tf.write(f"{Y[it]}\n")
        tf.close()
        tf_directory = os.path.dirname(tf.name)
    return run_aifeynman('', tf.name, BF_try_time=30,BF_ops_file_type='19ops.txt', polyfit_deg=4, NN_epochs=4000)[-1][-1]

# def gplearn_solve(X,Y):
#     m = SymbolicRegressor(generations=10 ** 1000)
#     m.fit(X,Y)
#     return str(m)

# def maysum_solve(X,Y):
#     m = Tree_Model()
#     m.train(np.array(X),np.array(Y))
#     return m.get_formula_string()

problems = mk_problems()
# random.shuffle(problems)

for it, problem in enumerate(problems):
    if it < 75: continue
    print(it, problem.eq_id,flush=True)
    start = time.time()
    try:
        ret = func_timeout(timeout=1000, func=feynman_solve, args=(problem.X,problem.Y))
        print( "SOLVED\n",ret, '\n vs \n', problem.form)
    except FunctionTimedOut:
        print("AI FEYNMAN TIMEOUT")
    except Exception as e:
        traceback.print_exc()
        print("AI FEYNMAN CRASHED")
    print('time', time.time() - start)
    print('====' * 25, flush=True)
