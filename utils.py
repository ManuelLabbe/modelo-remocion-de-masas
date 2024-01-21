import pandas as pd
import numpy as np
import seaborn as sns
import arviz as az
import pymc as pm
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pytensor
import pytensor.tensor as pt
from sklearn.metrics import (roc_curve, roc_auc_score, confusion_matrix, accuracy_score, f1_score, 
                             precision_recall_curve, classification_report, ConfusionMatrixDisplay,
                             recall_score, precision_score, classification_report)

floatX = pytensor.config.floatX

#path_slope_PP_vhs1 = 'data/processed/X_slope_PP_vhs1_factor09.csv'
class models_training_data:
    def __init__(self, PATH, train_size=0.8, random_state = 42):
        self.PATH = PATH
        self.train_size = train_size
        self.random_state = random_state
        self.X_train, self.X_test, self.y_train, self.y_test = self.extract_data()
    def extract_data(self):
        """_summary_

        Args:
            PATH (_type_): PATH de los datos
            train_size (float, optional): Conjunto de entrenamiendo. Defaults to 0.8.
            random_state (int, optional): Semilla Random. Defaults to 42.

        Returns:
            _type_: Conjunto de entenamiento y test: X_train, X_test, y_train, y_test
        """

        df = pd.read_csv(self.PATH)
        y = df.output
        X = df.drop(labels='output', axis=1)
        sc = StandardScaler()
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=self.train_size, stratify=y,random_state=self.random_state)
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        return X_train, X_test, y_train, y_test

    #X_train, X_test, y_train, y_test = extract_data(path_slope_PP_vhs1)

    def construct_nn(self,ann_input, ann_output):
        """_summary_

        Args:
            ann_input (_type_): Conjunto de entrenamiento X
            ann_output (_type_): Conjunto de entrenamiento Y

        Returns:
            _type_: Modelo de red neuronal bayesiana
        """
        n_hidden = 5
        random_seed = 12
        rng = np.random.default_rng(random_seed)
        # Initialize random weights between each layer
        init_1 = rng.standard_normal(size=(self.X_train.shape[1], n_hidden)).astype(floatX)
        init_2 = rng.standard_normal(size=(n_hidden, n_hidden)).astype(floatX)
        init_out = rng.standard_normal(size=n_hidden).astype(floatX)

        coords = {
            "hidden_layer_1": np.arange(n_hidden),
            "hidden_layer_2": np.arange(n_hidden),
            "train_cols": np.arange(self.X_train.shape[1]),
            # "obs_id": np.arange(X_train.shape[0]),
        }
        with pm.Model(coords=coords) as neural_network:
            ann_input = pm.Data("ann_input", self.X_train, mutable=True, dims=("obs_id", "train_cols"))
            ann_output = pm.Data("ann_output", self.y_train, mutable=True, dims="obs_id")

            # Weights from input to hidden layer
            weights_in_1 = pm.Normal(
                "w_in_1", 0, sigma=1, initval=init_1, dims=("train_cols", "hidden_layer_1")
            )

            # Weights from 1st to 2nd layer
            weights_1_2 = pm.Normal(
                "w_1_2", 0, sigma=1, initval=init_2, dims=("hidden_layer_1", "hidden_layer_2")
            )

            # Weights from hidden layer to output
            weights_2_out = pm.Normal("w_2_out", 0, sigma=1, initval=init_out, dims="hidden_layer_2")

            # Build neural-network using tanh activation function
            act_1 = pm.math.tanh(pm.math.dot(ann_input, weights_in_1))
            act_2 = pm.math.tanh(pm.math.dot(act_1, weights_1_2))
            act_out = pm.math.sigmoid(pm.math.dot(act_2, weights_2_out))

            # Binary classification -> Bernoulli likelihood
            out = pm.Bernoulli(
                "out",
                act_out,
                observed=ann_output,
                total_size=self.y_train.shape[0],  # IMPORTANT for minibatches
                dims="obs_id",
            )
        return neural_network

    def bayesian_model_w_factors(self,input, output):
        """_summary_

        Args:
            input (_type_): Conjunto de entrenamiento X
            output (_type_): Conjunto de entrenamiento y

        Returns:
            _type_: Modelo bayesiano con factores de grado superior
        """
        coords = {
            'train_cols': np.arange(self.X_train.shape[1]),
            #'obs_id': np.arange(X_train.shape[0])
        }
        with pm.Model(coords=coords) as model:
            input = pm.Data('input', self.X_train, mutable=True, dims=('obs_id','train_cols'))
            observed = pm.Data('observed', self.y_train, mutable=True, dims='dims_id')
            intercept = pm.Normal('intercept', mu = 0, sigma = 5) #5
            beta = pm.Normal('beta',mu=0, sigma=5)
            beta2 = pm.Normal('beta2', mu=0, sigma=5)
            beta3 = pm.Normal('beta3', mu = 0, sigma=5)
            alpha = pm.Normal('alpha', mu = 0, sigma=0.1)
            alpha2 = pm.Normal('alpha2', mu = 0, sigma = 0.1) #0.1
            alpha3 = pm.Normal('alpha3', mu = 0, sigma=1)
            alpha4 = pm.Normal('alpha4', mu = 0, sigma = 0.1) #0.1
            alpha5 = pm.Normal('alpha5', mu = 0 , sigma = 1) #1
            alpha6 = pm.Normal('alpha6', mu = 0 , sigma = 1) #1
            p = pm.Deterministic(
                'p', 
                pm.math.invlogit(
                    intercept + input[:,0]*beta + input[:,1]*beta2 + input[:,2]*beta3 + 
                    (input[:,1]**2)*alpha**2 + (input[:,2]**2)*alpha2**2 + (input[:,0]**2)*alpha3**2 + 
                    (input[:,1]**3)*alpha4**3 + (input[:,2]**3)*alpha5**3 + (input[:,0]**3)*alpha6**3))
            final = pm.Bernoulli('out',p , observed = observed, dims='obs_id')
            
        return model

    def simple_bayesian_model(self,input, output):
        """_summary_

        Args:
            input (_type_): Conjunto de entrenamiento X
            output (_type_): Conjunto de entrenamiento y

        Returns:
            _type_: Modelo bayesiano de grado 1
        """
        
        coords = {
            'train_cols': np.arange(self.X_train.shape[1]),
            #'obs_id': np.arange(X_train.shape[0])
        }
        with pm.Model(coords=coords) as model:
            input = pm.Data('input', self.X_train, mutable=True,)#dims=('obs_id','train_cols'))
            observed = pm.Data('observed', self.y_train, mutable=True,)#dims='dims_id')
            intercept = pm.Normal('intercept', mu=0,sigma=5)
        
            beta_1 = pm.Normal('beta_1',mu=0, sigma=5)
            beta_2 = pm.Normal('beta_2', mu=0, sigma=5)
            beta_3 = pm.Normal('beta_3', mu = 0, sigma=5)
            p = pm.Deterministic(
                'p', 
                pm.math.invlogit(
                    intercept + input[:,0]*beta_1 + input[:,1]*beta_2 + input[:,2]*beta_3
                ))
                
            out = pm.Bernoulli('out', p, observed = observed,)# dims='obs_id')
        return model

    def train_model(model):
        """_summary_

        Args:
            model (_type_): Modelo bayesiano grado 1 o superior

        Returns:
            _type_: trace post entremiento para predicción
        """
        with model:
            start = pm.find_MAP()
            #hessian_inv = np.linalg.inv(pm.find_hessian(start, model=bayesian_regression_tree))
            approx = pm.fit(n=10_000, start=start)
            step = pm.Metropolis()
            trace = approx.sample(draws=5000,)#step = step, start = start)# nuts_sampler_kwargs={'hess_inv': hessian_inv})
        return trace

    def train_nn_model(model):
        """_summary_

        Args:
            model (_type_): Modelo red neuronal bayesiano

        Returns:
            _type_: trace post entrenamiento para predicción
        """
        with model:
            approx = pm.fit(n=20_000)
        trace = approx.sample(draws=5000)
        return trace
        
    def model_inference(self,model, trace):
        """_summary_

        Args:
            model (_type_): Modelo de grado 1 o superior
            trace (_type_): trace post entrenamiento

        Returns:
            _type_: valores predichos, trace
        """
        with model:
            pm.set_data(new_data={"input": self.X_test})
            ppc = pm.sample_posterior_predictive(trace, var_names=['p'], return_inferencedata=True, predictions=True, extend_inferencedata=True)
            trace.extend(ppc)
        #pred = ppc.posterior_predictive["out"].mean(("chain", "draw")) > 0.50
        #print(f"Accuracy = {(y_test == pred.values).mean() * 100}%")
        pred = ppc.predictions['p'].mean(('chain', 'draw')) > 0.5
        print(f"Accuracy = {(self.y_test == pred.values).mean() * 100}%")
        return pred.values, trace 

    def nn_model_inference(self,model, trace):
        """_summary_

        Args:
            model (_type_): Modelo red neuronal bayesiana
            trace (_type_): trace post entrenaminto

        Returns:
            _type_: valores predichos, trace
        """
        with model:
            pm.set_data(new_data={"ann_input": self.X_test})
            ppc = pm.sample_posterior_predictive(trace)
            trace.extend(ppc)

        pred = ppc.posterior_predictive["out"].mean(("chain", "draw")) > 0.50
        print(f"Accuracy = {(self.y_test == pred.values).mean() * 100}%")
        return pred.values, trace
        
    def save_model(model, name_trace):

        with model:
            start = pm.find_MAP()
            #hessian_inv = np.linalg.inv(pm.find_hessian(start, model=bayesian_regression_tree))
            approx = pm.fit(n=10_000, start=start)
            step = pm.Metropolis()
            trace = approx.sample(draws=5000,)#step = step, start = start)# nuts_sampler_kwargs={'hess_inv': hessian_inv})

        with open(f'{name_trace}.pkl', 'wb') as f:
            pickle.dump(trace, f)
            
    def save_model_nn(model, name_trace):
        
        with model:
            approx = pm.fit(n=20_000)
        trace = approx.sample(draws=5000)

        with open(f'{name_trace}.pkl', 'wb') as f:
            pickle.dump(trace, f)
            
    def read_model(self,PATH):
        """_summary_

        Args:
            PATH (_type_): PATH donde está el modelo guardado (modelos/)

        Returns:
            _type_: return trace post entrenamiento del modelo
        """
        with open(f'{PATH}','rb') as f_trace:
            trace = pickle.load(f_trace)
            return trace

def metricas(y_test, y_score):
    """_summary_

    Args:
        y_test (_type_): y real
        y_score (_type_): y predicha
    
    Returns:
        _type_: return imprime metricas recall, presicion y matriz de confusión
    """
    recall = recall_score(y_test, y_score)
    precision = precision_score(y_test,y_score)
    print(f'RECALL:{recall: .3f}\nPRECISION:{precision: .3f}')
    confusion_matrix_ = confusion_matrix(y_test, y_score)
    disp = ConfusionMatrixDisplay(confusion_matrix_)
    disp.plot()
    plt.show()

def curvas_metricas(y_test, y_score):
    """_summary_

    Args:
        y_test (_type_): y real
        y_score (_type_): y predicho
        
    Returns:
        Imprime las curva roc, precision vs recall y f1 score
    """
    pred_scores = dict(y_true=y_test, y_score=y_score)
    cols = ['False Positive Rate', 'True Positive Rate', 'threshold']
    roc = pd.DataFrame(dict(zip(cols, roc_curve(**pred_scores))))
    precision, recall, ts = precision_recall_curve(y_true=y_test, probas_pred=y_score)
    pr_curve = pd.DataFrame({'Precision': precision, 'Recall': recall})
    f1 = pd.Series({t: f1_score(y_true=y_test, y_pred=y_score>t) for t in ts})
    best_threshold = f1.idxmax()
    fig, axes = plt.subplots(ncols=3, figsize=(13, 5))

    ax = sns.scatterplot(x='False Positive Rate', y='True Positive Rate', data=roc, s=50, legend=False, ax=axes[0])
    axes[0].plot('False Positive Rate', 'True Positive Rate', data=roc, lw=1, color='k')
    axes[0].plot(np.linspace(0,1,100), np.linspace(0,1,100), color='k', ls='--', lw=1)
    axes[0].fill_between(y1=roc['True Positive Rate'], x=roc['False Positive Rate'], alpha=.3, color='red')
    axes[0].set_title('Receiver Operating Characteristic')


    sns.scatterplot(x='Recall', y='Precision', data=pr_curve, ax=axes[1])
    axes[1].set_ylim(0,1)
    axes[1].set_title('Precision-Recall Curve')


    f1.plot(ax=axes[2], title='F1 Scores', ylim=(0,1))
    axes[2].set_xlabel('Threshold')
    axes[2].axvline(best_threshold, lw=1, ls='--', color='k')
    #axes[2].text(text=f'Max F1 @ {best_threshold:.2f}', x=.60, y=.95, s=5)
    fig.suptitle(f'roc_auc_score = {round(roc_auc_score(**pred_scores),2)}', fontsize=24)
    fig.tight_layout()
    plt.subplots_adjust(top=.8)
    plt.show();
    
class test():
    def __init__(self, a):
        self.a = a
    def square(self):
        return self.a**2
        