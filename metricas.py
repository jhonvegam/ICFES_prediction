from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve

def metricas_modelo(test, pred, nombre_mod):
    print("\n------------------- MÉTRICAS DEL MODELO -------------------------")
    print("\nAciertos: ", accuracy_score(test.loc[:,"target"], pred, normalize=False)) 
    print("\nTasa de Aciertos: ", accuracy_score(test.loc[:,"target"], pred)) 
    print("\nMatriz de confusion: ", confusion_matrix(test.loc[:,"target"], pred, labels=["Low", "Medium", "High", "Very High"]).ravel()) 

    #-------- Sensibilidad ------------------------------------------------------------------------------------------------------------
    from sklearn.metrics import recall_score

    recall = recall_score(test.loc[:, "target"], pred, labels=["Low", "Medium", "High", "Very High"], average=None)
    print("\nSensibilidad (Recall): ", recall)


    #-------- Exactitud ---------------------------------------------------------------------------------------------------------------

    from sklearn.metrics import precision_score

    precision = precision_score(test.loc[:, "target"], pred, labels=["Low", "Medium", "High", "Very High"], average=None)
    print("\nExactitud (Precision): ", precision)

    # # ------- CURVA ROC ----------------------------------------------------------------------------------------------------------------

    # # Crear un diccionario de mapeo de etiquetas a valores numéricos
    # label_mapping = {"Graduate": 1, "Enrolled": 0, "Dropout": 0}

    # # Aplicar el mapeo para convertir las etiquetas en valores numéricos
    # pred_numeric = [label_mapping[label] for label in pred]
    # test_numeric = [label_mapping[label] for label in test.loc[:, "target"]]

    # #ROC
    # from sklearn.metrics import roc_curve, roc_auc_score
    # import matplotlib.pyplot as plt

    # # Calcular la puntuación AUC
    # #auc = roc_auc_score(test.loc[:, "target"], pred)

    # # Ahora, puedes calcular el puntaje ROC AUC utilizando pred_numeric
    # auc = roc_auc_score(test_numeric, pred_numeric, multi_class='ovr')

    # # Calcular la curva ROC
    # fpr, tpr, _ = roc_curve(test_numeric, pred_numeric, pos_label=1)

    # # Graficar la curva ROC
    # plt.figure()
    # plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f})')
    # plt.xlabel('Tasa de Falsos Positivos (FPR)')
    # plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    # plt.title(f"Curva ROC - Modelo {nombre_mod}")
    # plt.legend(loc='lower right')
    # plt.show()
