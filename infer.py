from pgmpy.inference import VariableElimination

#Predicciones

#Función que genera el vector de la predicción
def inferencia(modelo, test):

    infer = VariableElimination(modelo)

    pred = []

    for i in range(len(test)):

        evidencias = {}

        for j in range(len(test.columns)):
            if test.columns[j] != "target":
                evidencias[test.columns[j]] = test.iloc[i][j]

        pred_test = infer.map_query(["target"], 
                            evidence=evidencias, show_progress=False)
        
        pred.append(pred_test["target"])

    return pred
 

#Función que pasa la predicción según los valores introducidos en el dash
def prediccion_dash_infer(modelo, ve ):

    infer = VariableElimination(modelo)

    pred_test = infer.map_query(["target"], 
                            evidence={"course": ve[0], "daytime/evening attendance": ve[1], "previous qualification (grade)": ve[2], 
                                        "displaced":ve[3], "tuition fees up to date": ve[4], "scholarship holder": ve[5], 
                                        "curricular units 1st sem (evaluations)": ve[6], "curricular units 1st sem (grade)":ve[7],
                                        "unemployment rate":ve[8], "inflation rate":ve[9],
                                        "gdp":ve[10]}, show_progress=False)
    return pred_test
