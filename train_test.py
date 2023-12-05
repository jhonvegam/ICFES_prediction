import pandas as pd
from pgmpy.models import BayesianNetwork
from sklearn.model_selection import train_test_split
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

from infer import inferencia, prediccion_dash_infer
import metricas as m

#Lectura de datos
df = pd.read_csv("data_discreta.csv", header = 0, index_col=0, sep=";")
df = df.astype('category')

#División entre Train y Test
train, test = train_test_split(df, test_size=0.2, random_state=101)

train.to_csv("train.csv", sep = ";", index = False)
test.to_csv("test.csv", sep = ";", index = False)
