import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

from despliegue.infer import inferencia
from despliegue.infer import prediccion_dash_infer
import metricas as m


#Lectura de datos
df = pd.read_csv("train.csv", sep = ";")
test = pd.read_csv("test.csv", sep = ";")


def probar_grafo(grafo, nombre):
    mod_fit_mv= BayesianNetwork(grafo)
    
    df_fit = df[list( mod_fit_mv.nodes())]


    #Modulo de ajuste para algunas CPDs del nuevo modelo
    emv = MaximumLikelihoodEstimator(model=mod_fit_mv, data=df_fit)

    #Parámetros obtenidos con la estumación de Máxima verosimilitud
    mod_fit_mv.fit(data=df_fit, estimator = MaximumLikelihoodEstimator) 
    #for i in mod_fit_mv.nodes():
        #print(mod_fit_mv.get_cpds(i)) 

    #Modelo de inferencia
    infere = VariableElimination(mod_fit_mv)
    test_fit = test[list(mod_fit_mv.nodes())]

    pred = inferencia(mod_fit_mv, test_fit)

    m.metricas_modelo(test_fit, pred, "Hibrido")

    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #Modelo con PUNTAJES - BIC para el K2
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    from pgmpy.metrics import structure_score

    print("\n* MODELO PC " + nombre + " - PUNTAJE: BIC \n")
    print(structure_score(mod_fit_mv, df_fit, scoring_method="bic"))

    print("\n* MODELO PC" + nombre + " - PUNTAJE: K2 \n")
    print(structure_score(mod_fit_mv, df_fit, scoring_method="k2"))

    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # SERIALIZACIÓN
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    import pickle

    filename="serializacion/modelo-" + nombre + ".pkl"
    with open(filename,'wb') as file:
        pickle.dump(mod_fit_mv, file)
        file.close()


#Modelo con estructura inicial sin parámetros
grafo1 = [('fami_tieneautomovil', 'fami_tieneinternet'), ('fami_tieneinternet', 'fami_tienecomputador'), ('fami_tieneinternet', 'fami_estratovivienda'), ('estu_edad_cat', 'cole_jornada'), ('cole_jornada', 'target'), ('cole_naturaleza', 'target'), ('cole_naturaleza', 'cole_calendario'), ('cole_naturaleza', 'desemp_ingles'), ('cole_naturaleza', 'cole_jornada'), ('desemp_ingles', 'fami_tieneinternet'), ('desemp_ingles', 'target'), ('desemp_ingles', 'cole_calendario'), ('desemp_ingles', 'fami_tienecomputador'), ('desemp_ingles', 'cole_jornada'), ('fami_tienecomputador', 'target'), ('fami_tienecomputador', 'fami_estratovivienda'), ('cole_bilingue', 'cole_calendario'), ('cole_bilingue', 'desemp_ingles')]
grafo2 = [('cole_bilingue','desemp_ingles'),('cole_calendario','cole_bilingue'),('cole_calendario','desemp_ingles'),('cole_calendario','cole_jornada'),('desemp_ingles','fami_estratovivienda'),('desemp_ingles','target'),('desemp_ingles','cole_jornada'),('cole_jornada','target'),('cole_jornada','fami_tieneautomovil'),('fami_tieneautomovil','fami_estratovivienda'),('cole_jornada','estu_edad_cat')]
probar_grafo(grafo1, "Hibrido-1")
probar_grafo(grafo2, "Hibrido-2")




