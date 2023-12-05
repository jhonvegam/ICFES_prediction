import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

from infer import inferencia
from infer import prediccion_dash_infer
import metricas as m


#Lectura de datos
df = pd.read_csv("train.csv", sep = ";")
test = pd.read_csv("test.csv", sep = ";")


def probar_grafo(grafo):
    mod_fit_mv= BayesianNetwork(grafo)


    #Modulo de ajuste para algunas CPDs del nuevo modelo
    emv = MaximumLikelihoodEstimator(model=mod_fit_mv, data=df)

    #Parámetros obtenidos con la estumación de Máxima verosimilitud
    mod_fit_mv.fit(data=df, estimator = MaximumLikelihoodEstimator) 
    for i in mod_fit_mv.nodes():
        print(mod_fit_mv.get_cpds(i)) 

    #Modelo de inferencia
    infer = VariableElimination(mod_fit_mv)
    test_fit = test[list(mod_fit_mv.nodes())]

    pred = inferencia(mod_fit_mv, test_fit)

    m.metricas_modelo(test_fit, pred, "K2")


#Modelo con estructura inicial sin parámetros
grafo1 = [('fami_tieneautomovil', 'fami_tieneinternet'), ('fami_tieneinternet', 'fami_tienecomputador'), ('fami_tieneinternet', 'fami_estratovivienda'), ('estu_edad_cat', 'cole_jornada'), ('cole_jornada', 'target'), ('cole_naturaleza', 'target'), ('cole_naturaleza', 'cole_calendario'), ('cole_naturaleza', 'desemp_ingles'), ('cole_naturaleza', 'cole_jornada'), ('desemp_ingles', 'fami_tieneinternet'), ('desemp_ingles', 'target'), ('desemp_ingles', 'cole_calendario'), ('desemp_ingles', 'fami_tienecomputador'), ('desemp_ingles', 'cole_jornada'), ('fami_tienecomputador', 'target'), ('fami_tienecomputador', 'fami_estratovivienda'), ('cole_bilingue', 'cole_calendario'), ('cole_bilingue', 'desemp_ingles')]
grafo2 = [('cole_naturaleza', 'cole_jornada'), ('cole_naturaleza', 'cole_calendario'), ('cole_naturaleza', 'fami_tienecomputador'), ('cole_calendario', 'desemp_ingles'), ('cole_calendario', 'cole_bilingue'), ('cole_jornada', 'desemp_ingles'), ('cole_jornada', 'cole_bilingue'), ('cole_jornada', 'cole_calendario'), ('cole_jornada', 'target'), ('fami_estratovivienda', 'cole_naturaleza'), ('fami_estratovivienda', 'fami_tienecomputador'), ('fami_estratovivienda', 'fami_tieneinternet'), ('fami_estratovivienda', 'estu_edad_cat'), ('fami_tieneinternet', 'fami_tieneautomovil'), ('fami_tienecomputador', 'fami_tieneinternet'), ('fami_tienecomputador', 'fami_tieneautomovil'), ('estu_edad_cat', 'cole_jornada'), ('estu_edad_cat', 'fami_tieneautomovil'), ('desemp_ingles', 'target'), ('desemp_ingles', 'fami_tienecomputador'), ('desemp_ingles', 'fami_tieneinternet'), ("fami_estratovivienda","fami_tienecomputador"),("fami_estratovivienda","fami_tieneinternet")]

probar_grafo(grafo1)
probar_grafo(grafo2)




