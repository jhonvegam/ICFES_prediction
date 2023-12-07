import dash
from dash import dcc  # dash core components
from dash import html # dash html components
from dash.dependencies import Input, Output
import requests
import json
from loguru import logger

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# app server
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

# PREDICTION API URL 
api_url = "http://44.211.196.239:8001/api/v1/predict"

# Layout in HTML
app.layout = html.Div(
    [
    html.H6("Ingrese la información del cliente:"),
    html.Div(["Número de meses inactivo en los los últimos 12 meses: ",
              dcc.Dropdown(id='inactivo', value='1', options=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12' ])]),
    html.Br(),
    html.Div(["Número de servicios toamdos por el cliente: ",
              dcc.Dropdown(id='servicios', value='1', options=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])]),
    html.Br(),
    html.Div(["Valor total de las transacciones: ",
              dcc.Input(id='transac', value='10', type='number')]),
    html.Br(),
    html.Div(["Número de transacciones: ",
              dcc.Input(id='num_transac', value='10')]),
    html.Br(),
    
    html.H6(html.Div(id='resultado')),
    
    ]
)

# Method to update prediction
@app.callback(
    Output(component_id='resultado', component_property='children'),
    [Input(component_id='inactivo', component_property='value'), 
     Input(component_id='servicios', component_property='value'), 
     Input(component_id='transac', component_property='value'), 
     Input(component_id='num_transac', component_property='value')]
)
def update_output_div(inactivo, servicios, transac, num_transac ):
    myreq = {
        "inputs": [
            {
            "Customer_Age": 57,
            "Gender": "M",
            "Dependent_count": 4,
            "Education_Level": "Graduate",
            "Marital_Status": "Single",
            "Income_Category": "$120K +",
            "Card_Category": "Blue",
            "Months_on_book": 52,
            "Total_Relationship_Count": int(servicios),
            "Months_Inactive_12_mon": int(inactivo),
            "Contacts_Count_12_mon": 2,
            "Credit_Limit": 25808,
            "Total_Revolving_Bal": 0,
            "Avg_Open_To_Buy": 25808,
            "Total_Amt_Chng_Q4_Q1": 0.712,
            "Total_Trans_Amt": int(transac),
            "Total_Trans_Ct": int(num_transac),
            "Total_Ct_Chng_Q4_Q1": 0.843,
            "Avg_Utilization_Ratio": 0
            }
        ]
      }
    headers =  {"Content-Type":"application/json", "accept": "application/json"}

    # POST call to the API
    response = requests.post(api_url, data=json.dumps(myreq), headers=headers)
    data = response.json()
    logger.info("Response: {}".format(data))

    # Pick result to return from json format
    result = "ALTO riesgo de abandono" if round(data["predictions"][0])==1 else "BAJO riesgo de abandono"
    
    return result 

 

if __name__ == '__main__':
    logger.info("Running dash")
    app.run_server(debug=True)
