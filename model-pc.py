#---------------------------------------------------------------------------------------
# Estimando la estructura de un modelo a partir de datos: puntajes
#---------------------------------------------------------------------------------------
import pandas as pd
from despliegue.infer import prediccion_dash_infer
from despliegue.infer import inferencia
import metricas as m

from pgmpy.estimators import PC

df = pd.read_csv("train.csv", sep = ";")
test = pd.read_csv("test.csv", sep = ";")

df = df.drop(df.columns[0], axis = 1 )

est = PC(data=df)

estimated_model = est.estimate(variant="stable", max_cond_vars=4)

print("\n* DAG\n")
print(estimated_model)

print("\n* Nodos del modelo (variables)\n")
print(estimated_model.nodes())

print("\n* Arcos del modelo (relaciones)\n")
print(estimated_model.edges())

grafo = [("cole_bilingue","desemp_ingles"),("cole_calendario","cole_bilingue"),("cole_calendario","desemp_ingles"),("cole_calendario","cole_jornada"),("desemp_ingles","target"),("desemp_ingles","fami_estratovivienda"),("fami_estratovivienda","fami_tieneinternet"),("fami_tieneautomovil","fami_estratovivienda"),("cole_jornada","fami_estratovivienda"),("cole_jornada","fami_tieneinternet"),("cole_jornada","estu_edad_cat"),("cole_jornada","target"),("fami_tieneinternet","fami_tienecomputador")]
#grafo = [('fami_tieneautomovil', 'fami_tieneinternet'), ('fami_tieneinternet', 'fami_tienecomputador'), ('fami_tieneinternet', 'fami_estratovivienda'), ('estu_edad_cat', 'cole_jornada'), ('cole_jornada', 'target'), ('cole_naturaleza', 'target'), ('cole_naturaleza', 'cole_calendario'), ('cole_naturaleza', 'desemp_ingles'), ('cole_naturaleza', 'cole_jornada'), ('desemp_ingles', 'fami_tieneinternet'), ('desemp_ingles', 'target'), ('desemp_ingles', 'cole_calendario'), ('desemp_ingles', 'fami_tienecomputador'), ('desemp_ingles', 'cole_jornada'), ('fami_tienecomputador', 'target'), ('fami_tienecomputador', 'fami_estratovivienda'), ('cole_bilingue', 'cole_calendario'), ('cole_bilingue', 'desemp_ingles')]

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

from pgmpy.models import BayesianNetwork
modelo = BayesianNetwork(grafo)

df_fit = df[list(estimated_model.nodes())]

from pgmpy.estimators import MaximumLikelihoodEstimator

emv = MaximumLikelihoodEstimator(model=modelo, data=df_fit)
modelo.fit(data=df_fit, estimator = MaximumLikelihoodEstimator) 

from pgmpy.inference import VariableElimination

#Modelo de inferencia
infer = VariableElimination(modelo)

test_fit = test[list(estimated_model.nodes())]


pred = inferencia(modelo, test_fit)

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Metricas del modelo predictivo
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

m.metricas_modelo(test_fit, pred, "PC")

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Modelo con PUNTAJES - BIC para el K2
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
from pgmpy.metrics import structure_score

print("\n* MODELO PC - PUNTAJE: BIC \n")
print(structure_score(modelo, df_fit, scoring_method="bic"))

print("\n* MODELO PC - PUNTAJE: K2 \n")
print(structure_score(modelo, df_fit, scoring_method="k2"))


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# SERIALIZACIÓN
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
import pickle

filename='serializacion/modelo-PC.pkl'
with open(filename,'wb') as file:
    pickle.dump(modelo, file)
    file.close()


