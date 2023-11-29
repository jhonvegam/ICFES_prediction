#---------------------------------------------------------------------------------------
# Estimando la estructura de un modelo a partir de datos: puntajes
#---------------------------------------------------------------------------------------
import pandas as pd
from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import K2Score

from infer import inferencia
import metricas as m

df = pd.read_csv("train.csv", sep = ";")
test = pd.read_csv("test.csv", sep = ";")

df = df.drop(df.columns[0], axis = 1 )

#-------------------------------------------------------------------------------------------------
# ESTIMACIÓN CON EL PUNTAJE K2
#-------------------------------------------------------------------------------------------------

scoring_method = K2Score(data=df)
esth = HillClimbSearch(data=df)
estimated_modelh = esth.estimate(
    scoring_method=scoring_method, max_indegree=4, max_iter=int(1e4)
)

print("\n* DAG\n")
print(estimated_modelh)

print("\n* Nodos del modelo (variables)\n")
print(estimated_modelh.nodes())

print("\n* Arcos del modelo (relaciones)\n")
print(estimated_modelh.edges())

print("\n* Puntaje\n")
print(scoring_method.score(estimated_modelh))

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

from pgmpy.models import BayesianNetwork
modelo = BayesianNetwork(list(estimated_modelh.edges()))

df_fit = df[list(estimated_modelh.nodes())]

from pgmpy.estimators import MaximumLikelihoodEstimator

emv = MaximumLikelihoodEstimator(model=modelo, data=df_fit)
modelo.fit(data=df_fit, estimator = MaximumLikelihoodEstimator) 

from pgmpy.inference import VariableElimination

#Modelo de inferencia
infer = VariableElimination(modelo)

test_fit = test[list(estimated_modelh.nodes())]

pred = inferencia(modelo, test_fit)

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Metricas del modelo predictivo
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

m.metricas_modelo(test_fit, pred, "K2")

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Modelo con PUNTAJES - BIC para el K2
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
from pgmpy.metrics import structure_score

print("\n* MODELO K2 - PUNTAJE: BIC \n")
print(structure_score(modelo, df_fit, scoring_method="bic"))

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# SERIALIZACIÓN
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
import pickle

filename='serializacion/modelo2-k2.pkl'
with open(filename,'wb') as file:
    pickle.dump(modelo, file)
    file.close()