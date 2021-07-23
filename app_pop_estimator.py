import pandas as pd
import numpy as np

### Reading data
pop = pd.read_csv(r"https://raw.githubusercontent.com/Alireza-Ehiaei/Data_Sciences/population_estim/Machine_learning/Deploy_on_Heroku/WPP2019_POP_F01_1_TOTAL_POPULATION_BOTH_SEXES.csv", encoding='latin-1')

#### Creating time series panel data 
def get_df_name(df):
    name =[x for x in globals() if globals()[x] is df][0]
    return name

def panel(df):
        
    name = get_df_name(df)
    df = df.drop(columns=['Index', 'Type','Variant','Notes', 'Country code','Parent code' ])
    
     # rmove all rows with NaN cells
    df= df.dropna(axis=0)
    
    df = df.rename(columns={'Region, subregion, country or area':'country' })
    # rows having '...' are not deleted, delete by:
    #df =df.loc[~((df['country']=='...') | (df['year']=='...') | (df['Region, subregion, country or area']=='...'))]    
    
    df['country'] = df['country'].replace({'Iran, Islamic Rep.': 'Iran'})
    dft = df.T
    new_header = dft.iloc[0]
    dft = dft[1:]
    dft.columns = new_header
    
    dft= dft.loc[:, ~(dft == '...').any()]
    
    for col in range(len(dft.columns)):
        dft.iloc[:,col] = dft.iloc[:,col].str.replace(' ','')
        
    return dft

panel = panel(pop)

### Definning function for estimating population for chosen countries for years [a,b]
# when year 0 is 1950
def autoreg(df,list_country,a,b):
    
    from statsmodels.tsa.ar_model import AutoReg
    from random import random
    #import pickle
    import json
    
    yhat =[]
    yhat= pd.DataFrame(yhat)
    # contrived dataset
    data = panel[panel.columns[panel.columns.isin(list_country)]]
    data = data.astype(int).reset_index(drop=True)
    # fit model
    for col in data.columns:
        model_AutoReg = AutoReg(data.loc[:,col], lags=1,old_names=False)
        model_fit = model_AutoReg.fit()
     
    #if there is just one model (one time series) save the model with pickle and 
    #call it in app:
    
    # pickle.dump(model_AutoReg , open('model.pkl','wb'))
    
    # make prediction
        yhat.loc[:,col] = model_fit.predict(a,b)
       

    # creating column year
    time = []
    year = []
    year = pd.DataFrame(year)
    time = pd.DataFrame(time)
    
    years = []
    years = range(a+1950,b+1951)
    yhat['years']= years 
    yhat =yhat.set_index(['years'])
    
    #removing scientific notation
    for col in yhat.columns:
        yhat.loc[:,col] = yhat.loc[:,col].apply(lambda x: '%.0f' % x +'000')
        # inserting the thousands separator
        yhat.loc[:,col] = yhat.loc[:,col].map(lambda x: f'{int(x):,}')
    
    return(yhat)


## Running the model using Flask

#for this app the index.html, layout.html and view.html files are saved in templates 
# folder and css file in static folder both whitin the main folder that includes this app(.py)  

from flask import Flask, flash, redirect, render_template, request, url_for
import json

app = Flask(__name__)
#we do not use pickle, becouse different variables would have different autoregression 
#models, otherwise one model could created by the cell above and call in this app by:
#model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template(
        'index.html',
        data= list(panel.columns))

@app.route("/test" , methods=['GET', 'POST'])
def test():
    
    list_ = request.form.getlist('countries')
    a = request.form.get('from') #starting year of time period
    b = request.form.get('to')   #ending year of time period
    
    data_etimate=autoreg(panel,list_,int(a)-1950,int(b)-1950)
    
    return render_template('view.html',tables=[data_etimate.to_html()],
    titles = ['na','Estimation of your selected data is:'])


if __name__=='__main__':
    #by using jupyter notebook use_teloser should ne false
    app.run(debug=True)
