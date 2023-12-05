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
        
        probabilidadesclases = infer.query(['target'], evidence=evidencias)

        pred.append(pred_test["target"])

    return pred
 
#Función que pasa la predicción según los valores introducidos en el dash
def prediccion_dash_infer(modelo, ve ):

    infer = VariableElimination(modelo)

    # 0 cole_naturaleza_dropdown', 'value'),                 -> NO
    # 1 State('cole_bilingue_dropdown', 'value'),
    # 2 State('cole_calendario_dropdown', 'value'),
    # 3 State('cole_jornada_dropdown', 'value'),
    # 4 State('fami_tieneautomovil_dropdown', 'value'),
    # 5 State('fami_estratovivienda_dropdown', 'value'),
    # 6 State('fami_tieneinternet_dropdown', 'value'),            -> NO
    # 7 State('fami_tienecomputador_dropdown', 'value'),            ->NO
    # 8 State('estu_edad_cat_dropdown', 'value'),
    # 9 State('desemp_ingles_dropdown')

    pred_test = infer.map_query(["target"], 
                            evidence={"cole_bilingue": ve[1], "cole_calendario": ve[2], "cole_jornada": ve[3], 
                                        "fami_tieneautomovil":ve[4], "fami_estratovivienda": ve[5], "fami_tieneinternet": ve[6],
                                        "fami_tienecomputador":ve[7],"estu_edad_cat":ve[8],
                                        "desemp_ingles":ve[9]}, show_progress=False)
    
    probabilidad = infer.query(['target'], evidence={"cole_bilingue": ve[1], "cole_calendario": ve[2], "cole_jornada": ve[3], 
                                        "fami_tieneautomovil":ve[4], "fami_estratovivienda": ve[5], "fami_tieneinternet": ve[6],
                                        "fami_tienecomputador":ve[7],"estu_edad_cat":ve[8],
                                        "desemp_ingles":ve[9]})

    print("cole_bilingue", ve[1])
    print("cole_calendario", ve[2])
    print("cole_jornada", ve[3])
    print("fami_tieneautomovil", ve[4])
    print("fami_estratovivienda", ve[5])
    print("fami_tieneautomovil", ve[6])
    print("fami_estratovivienda", ve[7])
    print("estu_edad_cat", ve[8])
    print("desemp_ingles", ve[9])


    return [pred_test["target"], max(probabilidad.values), probabilidad.values]
