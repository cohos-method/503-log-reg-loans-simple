import dash
from dash import dcc,html
import plotly.graph_objs as go
import pickle
import json
from dash.dependencies import Input, Output, State
from sklearn.pipeline import make_pipeline


myheading1 = "TEST"
tabtitle = "tabtitle"
fig = None
preops = [{'label': 'Top', 'value': 'top'}
        , {'label': 'High', 'value': 'high'}
        , {'label': 'Med', 'value': 'med'}
        , {'label': 'Normal', 'value': 'normal'}]
mdlops = [{'label': 'Logistic Regression'
        , 'value': "LR"}
        , {'label': 'Random Forest'
        , 'value': "RF"}
        , {'label': 'KNN'
        , 'value': "KNN"}]
githublink ="NONE"
sourceurl = "NONE"

########### Initiate the app
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.title=tabtitle


app.layout = html.Div(children=[
    html.H1(myheading1),
    html.Div([
        html.Div([
                html.Label('Choose your model:'),
                dcc.RadioItems(options=mdlops, value='LR',  id='mdl'),
                html.Label('Threshold:'),
                dcc.Graph(figure=fig, id='fig1')
                ], className='six columns'),
        html.Div([
                html.H3("Features"),
                html.Div('GPA (0.0 to 4.0):'),
                dcc.Input(id='gpa', value=0.8, type='number', min=0.0, max=1.0, step=0.01),
                html.Div('GRE:'),
                dcc.Input(id='gre', value=.8, type='number', min=0.0, max=1.0, step=.01),
                html.Div('Prestige Level (1-4)'),
                dcc.RadioItems(options=preops, value='high',  id='prestige'),
                html.Div('Probability Threshold for Admission'),
                dcc.Input(id='Threshold', value=50, type='number', min=0, max=100, step=1)
                ], className='three columns'),
        html.Div([
                html.H3('Predictions'),
                html.Div('Predicted Status:'),
                html.Div(id='PredResults'),
                html.Br(),
                html.Div('Probability of Approval:'),
                html.Div(id='ApprovalProb'),
                html.Br(),
                html.Div('Probability of Denial:'),
                html.Div(id='DenialProb')
                ], className='three columns')
        ], className='twelve columns'
    ),

    html.Br(),
    html.A('Code on Github', href=githublink),
    html.Br(),
    html.A("Data Source", href=sourceurl)
    ]
)

############ Deploy
if __name__ == '__main__':
    app.run_server(debug=True)
