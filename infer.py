from pgmpy.inference import VariableElimination

#Predicciones

#Función que genera el vector de la predicción
def inferencia(modelo, test):

    infer = VariableElimination(modelo)

    pred = []
    #prob = []

    for i in range(len(test)):

        evidencias = {}

        for j in range(len(test.columns)):
            if test.columns[j] != "target":
                evidencias[test.columns[j]] = test.iloc[i][j]

        pred_test = infer.map_query(["target"], 
                            evidence=evidencias, show_progress=False)
        
        #pred_test = infer.query(['target'], evidence=evidencias)
        probabilidadesclases = infer.query(['target'], evidence=evidencias)
        #probs = 
        print(max(probabilidadesclases.values))
        pred.append(pred_test["target"])
        #prob.append(pred_test)

    #print(prob[0])
    return pred
 
#Función que pasa la predicción según los valores introducidos en el dash
def prediccion_dash_infer(modelo, test, ve ):

    infer = VariableElimination(modelo)

    evidencias = {}

    for i in range(len(test.columns)):
        if test.columns[i] != "target":
            print(test.columns[i], ve[i])
            evidencias[test.columns[i]] = ve[i]

    pred_test = infer.map_query(["target"], evidence=evidencias, show_progress=False)
    
    probabilidad = infer.query(['target'], evidence=evidencias)

    return [pred_test["target"], max(probabilidad.values)]
