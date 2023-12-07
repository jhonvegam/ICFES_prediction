# -*- coding: utf-8 -*-

# Ejecute esta aplicación con 
# python app1.py
# y luego visite el sitio
# http://127.0.0.1:8050/ 
# en su navegador.

import dash
from dash import dcc  # dash core components
from dash import html # dash html components
from dash import State 
from dash.dependencies import Input, Output 
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import dash_mantine_components as dmc
# import bd_conexion as bd
import pickle
from infer import prediccion_dash_infer

#Read model from PKL file 
filename='docker/serializacion/modelo-PC.pkl'
file = open(filename, 'rb')
modelo = pickle.load(file)
file.close()

result =  pd.read_csv("data_discreta.csv", header = 0, index_col=0, sep=";")


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, 
                external_stylesheets=external_stylesheets, 
                suppress_callback_exceptions=True,
                title = 'Icfes - Secretaria Educación'
            )
server = app.server

def description_card():
    return html.Div(
        id="description-card",
        children=[
            html.H5("Icfes Análisis"),
            html.H3("Bienvenido al Dashboard de resultados de la prueba Saber 11"),
            html.Div(
                id="intro",
                children="Este Dashboard fue creado como parte de un proyecto con el objetivo de contribuir a la identificación de factores que influyen de manera positiva en los resultados de las pruebas saber 11. Esta herramienta fue creada para que cualquier persona o institución pueda hacer uso de los datos y plantear estrategias que optimicen los resultados",
            )
        ],
    )

def create_dropdown(id,options,value, label):
    return html.Div(
        children=[
            html.Label([label], style={'font-weight': 'bold', "text-align": "left"}),
            dcc.Dropdown(
                id=id,
                options=options,
                value=value
            )
        ]
    )

# Función para definir visualizaciones hist
def prediction_card():
    return html.Div(
                    id="prediction-card",
                    children = [
                        html.B("Mapa de calor puntaje global vs variable seleccionada (% del total de la fila)"),
                        html.Hr(),
                        dmc.Group(
                            position = 'center',
                            children = [
                                create_dropdown('visualization_hist_dropdown', result.columns, 'cole_naturaleza', 'Seleccione una columna para visualizar'),
                            ]
                        ),
                        html.Br(),
                        dmc.Group(
                            position = 'center',
                            children = [
                                
                                html.Div(
                                    id="reset-btn-hist",
                                    children=html.Button(id="btn-hist", children="Visualizar", n_clicks=0),
                                ),
                            ]
                        ),
                        dcc.Graph(id='target_heatmap'),
                        ],
                    ), html.Div(
                    id="other-graphs",
                    children = [
                        html.Br(),
                        html.B("Total de casos por variable seleccionada"),
                        html.Hr(),
                        dcc.Graph(id='hist-graph'),
                    ],
                    )

def prediction_dropdowns():  
    return html.Div(
            id="prediction-card",
            children = 
            [
                html.H2("Seleccione las características del colegio:"),
                html.Br(),
                dmc.SimpleGrid(
                            cols = 4,
                            children = [
                                create_dropdown('cole_naturaleza_dropdown', result.cole_naturaleza.unique(), 'OFICIAL', 'Tipo Colegio'),
                                create_dropdown('cole_bilingue_dropdown', result.cole_bilingue.unique(), 'No', 'Colegio Bilingue'),
                                create_dropdown('cole_calendario_dropdown', result.cole_calendario.unique(), 'A', 'Calendario'),
                                create_dropdown('cole_jornada_dropdown', result.cole_jornada.unique(), 'UNICA', 'Jornada'),
                                create_dropdown('fami_tieneautomovil_dropdown', result.fami_tieneautomovil.unique(), 'No', 'Automovil'),
                                create_dropdown('fami_estratovivienda_dropdown', result.fami_estratovivienda.unique(), 'Estrato 1 y 2', 'Estrato'),
                                create_dropdown('fami_tieneinternet_dropdown', result.fami_tieneinternet.unique(), 'No', 'Internet'),
                                create_dropdown('fami_tienecomputador_dropdown', result.fami_tienecomputador.unique(), 'No', 'Computador'),
                                create_dropdown('estu_edad_cat_dropdown', result.estu_edad_cat.unique(), '<= 25', 'Edad'),
                                create_dropdown('desemp_ingles_dropdown', result.desemp_ingles.unique(), 'A-', 'Nivel Ingles'),
                            ]
                ),
                html.Br(),
                dmc.Group(
                    position = 'center',
                    children = [
                        html.Div(
                            id="reset-btn-outer",
                            children=html.Button(id="reset-btn", children="Predecir", n_clicks=0), #  style= {'width':'50%','margin':'auto'}
                        ),
                    ]
                ),
                # Agregar prediccion
                dcc.Graph(id='speed_fig'),
            ],
        )

def generate_prediction_card():
    return html.Div(
        id="prediction-card2",
        children=[
            html.Br(),
            html.Div("Para generar una predicción y estimar el resultado de la prueba saber 11, por favor siga los siguientes pasos:"),
            html.Br(),
            html.Ol([
                html.Li("Use las listas desplegables que están en el panel de la derecha, para seleccionar las características de interés que se desean estudiar"),
                html.Li("Después de seleccionar los filtros deseados, asegúrese de que los campos hayan sido correctamente seleccionados y de click en el botón Predecir"),
                html.Li("Utilice la predicción generada para implementar estrategias apropiadas y medidas de apoyo para ayudar a los colegios o estudiantes a mejorar sus resultados.")
            ]),
        ],
    )

def visualizations_card():
    return html.Div(
        id="visualizations_card",
        children=[
            html.Br(),
            html.Div("Para visualizar el comportamiento de las variables vs el puntaje global, tenga en cuenta lo siguiente:"),
            html.Br(),
            html.Ol([
                html.Li("Use la listas desplegable que está en el panel de la derecha, para seleccionar la variable que desea visualizar"),
                html.Li("El mapa de calor muestra los porcentajes de casos para el total de la fila"),
                html.Li("Al seleccionar una casilla podrá filtrar y visualizar el total de casos en el gráfico de barras.")
            ]),
        ],
    )

def generate_heatmap (column, hm_click):  
    x_axis = ['Low', 'Medium', 'High', 'Very High']
    y_axis = result[column].unique().tolist()
    #region click heatmap
    target_value = ""
    grade_1st = ""
    shapes=[]
    if hm_click is not None:
        target_value = hm_click["points"][0]["x"]
        grade_1st = hm_click["points"][0]["y"]

        # Add shapes
        x0 = x_axis.index(target_value) / len(x_axis)
        x1 = x0 + 1 / len(x_axis)
        y0 = y_axis.index(grade_1st) / len(y_axis)
        y1 = y0 + 1 / len(y_axis)

        shapes = [
            dict(
                type="rect",
                xref="paper",
                yref="paper",
                x0=x0,
                x1=x1,
                y0=y0,
                y1=y1,
                line=dict(color="#ff6347"),
            )
        ]

    # Get z value : sum(number of records) based on x, y,
    z = np.zeros((len(y_axis), len(x_axis)))
    annotations = []

    for ind_y, row in enumerate(y_axis):
        filtered_row = result.loc[result[column] == row]
        total_schools = result[column].loc[result[column] == row].count()
        for ind_x, x_val in enumerate(x_axis):
            count_target = round(filtered_row[filtered_row["target"] == x_val]["target"].count()/total_schools * 100,2)
            value = str(count_target)
            z[ind_y][ind_x] = value

            annotation_dict = dict(
                showarrow=False,
                text="<b>" + value + "<b>",
                xref="x",
                yref="y",
                x=x_val,
                y=row,
                font=dict(family="sans-serif"),
            )
            # Highlight annotation text by self-click
            if x_val == target_value and row == grade_1st:
                annotation_dict.update(size=20, font=dict(color="#000000"))

            annotations.append(annotation_dict)
    #endregion

    # Heatmap
    hovertemplate = "<b> %{y}  %{x} <br><br> %{z} % Colegios por fila"

    data = [
        dict(
            x=x_axis,
            y=y_axis,
            z=z,
            type="heatmap",
            name="",
            hovertemplate=hovertemplate,
            showscale=False,
            colorscale=[[0, "#caf3ff"], [1, "#2c82ff"]],
        )
    ]

    layout = dict(
        margin=dict(l=70, b=50, t=50, r=50),
        modebar={"orientation": "v"},
        font=dict(family="Open Sans"),
        annotations=annotations,
        shapes=shapes,
        xaxis=dict(
            side="top",
            ticks="",
            ticklen=2,
            tickfont=dict(family="sans-serif"),
            tickcolor="#ffffff",
        ),
        yaxis=dict(
            side="left", ticks="", tickfont=dict(family="sans-serif"), ticksuffix=" "
        ),
        hovermode="closest",
        showlegend=False,
    )
    return {"data": data, "layout": layout}

#Función que pasa la predicción según los valores introducidos en el dash
def prediccion_dash(ve):

    return prediccion_dash_infer(modelo, ve)


app.layout =  html.Div(
    id="app-container",
    children= [
        # Banner
        html.Div(
            id="banner",
            className="banner",
            children=[html.Img(src=app.get_asset_url("logo_uniandes.png"))],
        ),
        # Left column
        html.Div( 
            id="left-column",
            className="three columns",
            children=[description_card(),
            dcc.Tabs(id='tabs', value='tab-v', children=[
                dcc.Tab(label='Visualizaciones', value='tab-v'),
                dcc.Tab(label='Predicción', value='tab-p'),
            ]),
            #region Update left column
            html.Div(id='tab_update_left_col')] 
            + [
                html.Div(
                    ["initial child"], id="output-clientside", style={"display": "none"}
                )
            ]
            #endregion
        ),
        # Right column
        html.Div(
            id="right-column",
            className="eight columns",
            children= [
            html.Div(id='tab_update_right_col')
                
            ]
        ),
])

# Tabs Callback
@app.callback(
    Output('tab_update_right_col', 'children'),
    Output('tab_update_left_col', 'children'),
    Input('tabs', 'value')
)
def render_content(tab):
    if tab == 'tab-v':
        return prediction_card(), visualizations_card()
    elif tab == 'tab-p':
        return prediction_dropdowns(), generate_prediction_card()


# Visualizaciones Callback
@app.callback(
            Output('hist-graph', 'figure'),
            Output('target_heatmap', 'figure'),
            Input('reset-btn-hist', "n_clicks"),
            State('visualization_hist_dropdown', 'value'),
            Input("target_heatmap", "clickData")
        )
def update_output(n_clicks, col_visualization, hm_click): 
    filtered_hm = result
    reset = False
    # Find which one has been triggered
    ctx = dash.callback_context

    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "reset-btn-hist":
            reset = True

    if reset:
        hm_click = None
    
    if hm_click is not None:
        filtered_hm = result.loc[
            (result["target"] == hm_click["points"][0]["x"])  & (result[col_visualization] == hm_click["points"][0]["y"])
        ]
    else:
        filtered_hm = result

    hist_plt = px.histogram(filtered_hm,
                            x=col_visualization, 
                            text_auto=True, 
                            color = "target", 
                            color_discrete_sequence= ["#4a9bff", "#6dbbff", "#90d9ff", "#b3e7ff"]).update_xaxes(categoryorder="total descending")
    hist_plt.update_layout(
        xaxis_title= col_visualization,
        yaxis_title='Total de colegios'
    )
    return hist_plt, generate_heatmap(col_visualization, hm_click) 

@app.callback( #Pendiente callback
    Output('speed_fig', 'figure'),
    Input("reset-btn", "n_clicks"),
    State('cole_naturaleza_dropdown', 'value'),
    State('cole_bilingue_dropdown', 'value'),
    State('cole_calendario_dropdown', 'value'),
    State('cole_jornada_dropdown', 'value'),
    State('fami_tieneautomovil_dropdown', 'value'),
    State('fami_estratovivienda_dropdown', 'value'),
    State('fami_tieneinternet_dropdown', 'value'),
    State('fami_tienecomputador_dropdown', 'value'),
    State('estu_edad_cat_dropdown', 'value'),
    State('desemp_ingles_dropdown', 'value'),
    prevent_initial_call=False
)

def update_output(n_clicks, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10):
    
    values = [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10]
    
    prediccion_resultado = prediccion_dash(values) # [clase, probabilidad]
   
    speed_fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prediccion_resultado[1]*100,
        number = {"suffix": "%"},
        title = {'text': prediccion_resultado[0]},
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {'axis': {'range': [None, 100]},
            'steps' : [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 100], 'color': "gray"}]
            }
    ))
    return speed_fig


if __name__ == '__main__':
    app.run_server(debug=True)



