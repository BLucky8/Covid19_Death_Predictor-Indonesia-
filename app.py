#import libraries
import numpy as np
import pandas as pd
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from flask import Flask, request, render_template
from pandas import to_datetime
from datetime import timedelta

#Initialize the flask App
app = Flask(__name__)
# model = pickle.load(open('model_pkl', 'rb'))

df = pd.read_excel("Clean.xlsx")

degree = 3
x= df[["Day"]].values.reshape(df[['Day']].size,1)
y= df[['Meninggal\nDunia']].values.reshape(df[['Meninggal\nDunia']].size,1)

polynomial_features= PolynomialFeatures(degree=degree)
x_poly = polynomial_features.fit_transform(x)

model = LinearRegression(normalize=True, fit_intercept=True)
model.fit(x_poly, y)
y_poly_pred = model.predict(x_poly)

#default page of our web-app
@app.route('/')
def home():
    return render_template('index.html')


def forecast_days_after(n_days_after):
    model_pred = model.predict(polynomial_features.fit_transform([[n_days_after]]))
    return '{:d}'.format(int(round(model_pred[0][0]))- 29230) 

start_date = to_datetime(df['Tanggal'].iloc[0], format='%m/%d/%Y')
last_date = to_datetime(df['Tanggal'].iloc[-1], format='%m/%d/%Y')

def days_passed_timedelta():
    last_date = to_datetime(df['Tanggal'].iloc[-1], format='%m/%d/%Y')
    return last_date.date() - start_date.date() 

def date_days_after(n_days_after, date_format="%d %B %Y", **kwargs):
    tanggal = start_date + timedelta(days=n_days_after)
    return tanggal.date().strftime(date_format)

#To use the predict button in our web-app
@app.route('/predict',methods=['POST'])
def predict():
    indexx = []
    tanggal1 = []
    prediksi1 = []
    selisih1 = []
    days_after = request.form.get('days_after')
    days_passed = days_passed_timedelta().days
    coba = forecast_days_after(days_passed+1)
    selisih1.append(0)
    indexx.append(days_passed+1)
    tanggal1.append(date_days_after(days_passed+1, date_format="%d-%m-%Y"))
    prediksi1.append(coba)
    n = 1
    z = 1
    for x in range(int(days_passed)+2,int(days_after)+2,n) :
        for y in range(n) :
          pengurangan  = 0+z 
          z += 1.3952
        mines = (700 * n)  + pengurangan
        n += 1
        prediksi = int(forecast_days_after(x)) - mines
        prediksi2 = (int(forecast_days_after(x+1)) - mines) -( 702 )
        tanggal = date_days_after(x, date_format="%d-%m-%Y")
        selisih = (prediksi2 - prediksi)
        indexx.append(x)
        tanggal1.append(tanggal)
        prediksi1.append(prediksi)
        selisih1.append(selisih)
        
    AA = {"Ke-":indexx , "Tanggal" : tanggal1 , "Prediksi" : prediksi1 , "Selisih" : selisih1}
    pred  = pd.DataFrame(AA)
    BB = pred.values.tolist()

    return render_template('index.html', BB = BB)
if __name__ == "__main__":
    app.run(debug=True)