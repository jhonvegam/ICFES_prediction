#---------------------------------------------------------------------------------------
# Estimando la estructura de un modelo a partir de datos: puntajes
#---------------------------------------------------------------------------------------
import pandas as pd
from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import BicScore

from despliegue.infer import inferencia
import metricas as m

df = pd.read_csv("train.csv", sep = ";")
test = pd.read_csv("test.csv", sep = ";")

#df = df.drop(df.columns[0], axis = 1 )
print(df.columns)

#-----------------------------------------------------------------------------------------------
# ESTIMACIÓN CON EL PUNTAJE BIC
#------------------------------------------------------------------------------------------------

scoring_method = BicScore(data=df)
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
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#grafo = list(estimated_modelh.edges())
#grafo = [('cole_naturaleza', 'cole_jornada'), ('cole_naturaleza', 'cole_calendario'), ('cole_naturaleza', 'fami_tienecomputador'), ('cole_calendario', 'desemp_ingles'), ('cole_calendario', 'cole_bilingue'), ('cole_jornada', 'desemp_ingles'), ('cole_jornada', 'cole_bilingue'), ('cole_jornada', 'cole_calendario'), ('cole_jornada', 'target'), ('fami_estratovivienda', 'cole_naturaleza'), ('fami_estratovivienda', 'fami_tienecomputador'), ('fami_estratovivienda', 'fami_tieneinternet'), ('fami_estratovivienda', 'estu_edad_cat'), ('fami_tieneinternet', 'fami_tieneautomovil'), ('fami_tienecomputador', 'fami_tieneinternet'), ('fami_tienecomputador', 'fami_tieneautomovil'), ('estu_edad_cat', 'cole_jornada'), ('estu_edad_cat', 'fami_tieneautomovil'), ('desemp_ingles', 'target'), ('desemp_ingles', 'fami_tienecomputador'), ('desemp_ingles', 'fami_tieneinternet'), ("fami_estratovivienda","fami_tienecomputador"),("fami_estratovivienda","fami_tieneinternet")]


from pgmpy.models import BayesianNetwork
modelo = BayesianNetwork(list(estimated_modelh.edges()))

nodos_fit = list(estimated_modelh.nodes())
#nodos_fit.pop(1) # -----> Tuve que eliminar la variable (previous qualification (grade)) porque estaba entre los nodos pero no se encontraba dentro de las relaciones y salia un error
#print(nodos_fit)

df_fit = df[nodos_fit]

from pgmpy.estimators import MaximumLikelihoodEstimator
emv = MaximumLikelihoodEstimator(model=modelo, data=df_fit)
modelo.fit(data=df_fit, estimator = MaximumLikelihoodEstimator) 

from pgmpy.inference import VariableElimination

#Modelo de inferencia
infer = VariableElimination(modelo)

test_fit = test[nodos_fit]

pred = inferencia(modelo, test_fit)

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Metricas del modelo predictivo
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

m.metricas_modelo(test_fit, pred, "BIC")

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Modelo con PUNTAJES - K2 para el BIC
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
from pgmpy.metrics import structure_score

print("\n* MODELO BIC - PUNTJAE: K2 \n")
print(structure_score(modelo, df_fit, scoring_method="k2"))

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# SERIALIZACIÓN
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
import pickle

filename='despliegue/serializacion/modelo2-bic.pkl'
with open(filename,'wb') as file:
    pickle.dump(modelo, file)
    file.close()

