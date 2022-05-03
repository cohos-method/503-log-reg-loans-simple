import dash
from dash import dcc,html
import plotly.graph_objs as go
import pickle
import json
import sklearn

from dash.dependencies import Input, Output, State
from sklearn import (
    metrics,
    linear_model,
    ensemble,
    neighbors,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
print ("\n\n\n*** Importing done...")

########### Define your variables ######
myheading1='College Admissio Predictions'
image1='assets/rocauc.html'
tabtitle = 'College Admission Prediction'
sourceurl = 'https://datahack.analyticsvidhya.com/contest/practice-problem-loan-prediction-iii/'
githublink = 'https://github.com/cohos-method/503-log-reg-loans-simple.git'
print ("*** 1. Init done...")

########### open the json file ######
with open('assets/rocauc.json', 'r') as f:
    fig=json.load(f)

print ("*** 2. Fig loaded...")

def unpickleme(fname):
    try:
        filename = open(fname, 'rb')
        obj = pickle.load(filename)
        filename.close()
        return obj
    except:
        return None

def buildOptionsDict(lbl, value):
    opt = []
    for i in range(len(lbl)):
        d = {}
        d['label'] = lbl[i]
        d['value'] = value[i]
        opt.append(d)
    return opt

preslbl = ['Top','High', 'Med', 'Normal']
preops = buildOptionsDict(preslbl, preslbl)
print ("*** 3. Building Options done...", preops)

mdllbl = ['Logistic Regression','Random Forest', 'KNN']
mdlops = buildOptionsDict(mdllbl, mdllbl)
print ("*** 4. Building Options done...", mdlops)

########### open the pickle file ######
flr = 'analysis/admission_logistic_model.pkl'
frf  = 'analysis/admission_forest_model.pkl'
fknn = 'analysis/admission_knn_model.pkl'

lr = unpickleme(flr)
rf = unpickleme(frf)
knn = unpickleme(fknn)
print ("*** 5. Unpickling done...")


########### Initiate the app
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.title=tabtitle

########### Set up the layout
app.layout = html.Div(children=[
    html.H1(myheading1),
    html.Div([
        html.Div([
                html.H3('Choose your model:'),
                dcc.RadioItems(options=mdlops, value='Logistic Regression',  id='mdl'),
                html.Br(),
                html.H3('Threshold:'),
                dcc.Graph(figure=fig, id='fig1')
                ], className='six columns'),
        html.Div([
                html.H3("Features"),
                html.Div('GPA (0.0 to 1.0):'),
                dcc.Input(id='gpa', value=0.8, type='number', min=0.0, max=1.0, step=0.01),
                html.Br(),
                html.Br(),
                html.Div('GRE (0.0 to 1.0):'),
                dcc.Input(id='gre', value=0.8, type='number', min=0.0, max=1.0, step=.01),
                html.Br(),
                html.Br(),
                html.Div('Prestige Level:'),
                dcc.RadioItems(options=preops, value='High',  id='prestige'),
                html.Br(),
                html.Br(),
                html.Div('Probability Threshold for Admission'),
                dcc.Input(id='Threshold', value=50, type='number', min=0, max=100, step=1)
                ], className='three columns'),
        html.Div([
                html.H3('Predictions'),
                html.H6(id="ModelName", children=""),
                html.Br(),
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


######### Define Callback
@app.callback(
    [Output(component_id='PredResults', component_property='children'),
     Output(component_id='ApprovalProb', component_property='children'),
     Output(component_id='DenialProb', component_property='children'),
     Output(component_id='ModelName', component_property='children'),
    ],
    [Input(component_id='mdl', component_property='value'),
     Input(component_id='gre', component_property='value'),
     Input(component_id='gpa', component_property='value'),
     Input(component_id='prestige', component_property='value'),
     Input(component_id='Threshold', component_property='value')
    ])
def prediction_function(vmdl, vgre, vgpa, vprestige, Threshold):
    try:
        print ("\n\n****** 7.0. GRE: {}, GPA: {}, Prestige: {}, Threshold: {}, MDL: {}".format(vgre, vgpa, vprestige, Threshold, vmdl))
        if vprestige == 'Top':
            dprestige = [1,0,0,0]

        elif vprestige == 'High':
            dprestige = [0,1,0,0]

        elif vprestige == 'Med':
            dprestige = [0,0,1,0]

        elif vprestige == 'Normal':
            dprestige = [0,0,0,1]

        else:
            dprestige = [1,0,0,0]


        if vmdl == "Logistic Regression":
            unpickled_model=lr
            mdlname= 'Logist Regression'

        elif vmdl == "Random Forest":
            unpickled_model=rf
            mdlname= 'Random Forest'

        elif vmdl == "KNN":
            unpickled_model=knn
            mdlname= 'K Nearest Neighbors'

        else:
            unpickled_model= None
            mdlname= 'Unknown'


        ls = []
        ls.append(vgre)
        ls.append(vgpa)
        ls= ls + dprestige
        print ("****** 7.1. ls:", ls)

        data = [ls]
        print ("****** 7.2. data (Features):", ls)

        print ("****** 7.3. Unpickled Model:", mdlname, unpickled_model)

        rawprob=100 * unpickled_model.predict_proba(data)[0][1]
        print ("****** 7.4. rawprob:",rawprob)

        func = lambda y: '*** ADMIT ***' if int(rawprob)>Threshold else '*** NO ADMISSION ***'
        formatted_y = func(rawprob)
        print ("****** 7.5. formatted_y:",formatted_y)

        deny_prob=unpickled_model.predict_proba(data)[0][0]*100
        print ("****** 7.6 deny_prob:",deny_prob)

        formatted_deny_prob = "{:,.2f}%".format(deny_prob)
        print ("****** 7.7. formatted_deny_prob:",formatted_deny_prob)

        app_prob=unpickled_model.predict_proba(data)[0][1]*100
        print ("****** 7.8. app_prob:",app_prob)

        formatted_app_prob = "{:,.2f}%".format(app_prob)
        print ("****** 7.9. formatted_app_prob:",formatted_app_prob)

        return formatted_y, formatted_app_prob, formatted_deny_prob, mdlname
    except:
        return "inadequate inputs", "inadequate inputs", "inadequate inputs", "unknown"





############ Deploy
if __name__ == '__main__':
    app.run_server(debug=True)
