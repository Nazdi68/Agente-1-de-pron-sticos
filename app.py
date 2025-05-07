import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression
from io import BytesIO

def detect_trend(y, threshold=0.01):
    t = np.arange(len(y))
    coef = np.polyfit(t, y, 1)[0]
    return abs(coef) > threshold, coef

def detect_seasonality(y, freq=12, thresh=0.01):
    dec = seasonal_decompose(y, model='additive', period=freq,
                             two_sided=False, extrapolate_trend='freq')
    return dec.seasonal.std() > thresh, dec.seasonal.std()

st.title("🧠 Agente de Pronósticos para Toma de Decisiones")

# 1) Carga de datos
st.sidebar.header("1) Carga tu serie de tiempo")
uploaded = st.sidebar.file_uploader("", type=['csv','xlsx'])
if uploaded:
    if uploaded.name.endswith('.csv'):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)
    df['Fecha'] = pd.to_datetime(df['Fecha'])
else:
    dates = pd.date_range('2022-01-01', periods=24, freq='M')
    sales = [200 + i*10 + 5*(-1)**i for i in range(24)]
    df = pd.DataFrame({'Fecha': dates, 'Ventas': sales})

st.subheader("Datos cargados")
st.dataframe(df)

# 2) Diagnóstico automático
y = df['Ventas'].values
has_trend, coef = detect_trend(y)
has_seas, seas_std = detect_seasonality(y)

st.markdown(f"- **Tendencia**: {has_trend} (coef = {coef:.3f})")
st.markdown(f"- **Estacionalidad**: {has_seas} (std = {seas_std:.3f})")

if not has_trend and not has_seas:
    suggestion = 'Simple'
elif has_trend and not has_seas:
    suggestion = 'Holt'
elif has_seas and not has_trend:
    suggestion = 'HW Aditivo'
else:
    suggestion = 'HW Multiplicativo'
st.markdown(f"✅ **Modelo sugerido:** {suggestion}")

# 3) Selección de modelo
models = [
    'Simple','Holt','HW Aditivo','HW Multiplicativo','ARIMA',
    'Regresión Lineal Simple','Regresión Lineal Múltiple'
]
model_name = st.sidebar.selectbox("2) Elige un modelo", models, index=models.index(suggestion))

# 4) Horizonte y pronóstico
n = len(y)
limits = {
    'Simple': int(n*0.25),'Holt': int(n*0.5),
    'HW Aditivo':24,'HW Multiplicativo':12,
    'ARIMA':n,'Regresión Lineal Simple':n,'Regresión Lineal Múltiple':n
}
hmax = limits[model_name]
h = st.sidebar.slider("3) Meses a pronosticar", 1, hmax, min(6,hmax))

if model_name == 'ARIMA':
    order = (1,1,1)
    seasonal = (1,1,1,12) if has_seas else (0,0,0,0)
    mod = SARIMAX(y, order=order, seasonal_order=seasonal,
                  enforce_stationarity=False, enforce_invertibility=False)
    fit = mod.fit(disp=False)
    pred = fit.get_forecast(steps=h).predicted_mean

elif model_name == 'Regresión Lineal Simple':
    t = np.arange(n).reshape(-1,1)
    lr = LinearRegression().fit(t, y)
    future_t = np.arange(n, n+h).reshape(-1,1)
    pred = lr.predict(future_t)

elif model_name == 'Regresión Lineal Múltiple':
    exog_cols = [c for c in df.columns if c not in ('Fecha','Ventas')]
    X = df[exog_cols].astype(float)
    lr = LinearRegression().fit(X, y)
    last = X.iloc[-1].values.reshape(1,-1)
    Xf = np.repeat(last, h, axis=0)
    pred = lr.predict(Xf)

else:
    params = {}
    if model_name == 'Holt': 
        params['trend']='add'
    if model_name == 'HW Aditivo':
        params.update({'seasonal':'add','seasonal_periods':12})
    if model_name == 'HW Multiplicativo':
        params.update({'trend':'add','seasonal':'mul','seasonal_periods':12})
    fit = ExponentialSmoothing(y, **params).fit()
    pred = fit.forecast(h)

future = [df['Fecha'].iloc[-1] + timedelta(days=30*(i+1)) for i in range(h)]
fc = pd.DataFrame({'Fecha': future, 'Pronóstico': pred})

# 5) Mostrar resultados
st.subheader("Pronóstico")
st.line_chart(fc.set_index('Fecha'))
st.dataframe(fc)

mean_diff = pred.mean() - df['Ventas'].tail(h).mean()
if mean_diff > 0:
    st.success("🔔 Tendencia al alza: aumenta inventario o expande recursos.")
else:
    st.warning("🔔 Tendencia a la baja: optimiza recursos o revisa costos.")

# 6) Generación de Excel
output = BytesIO()
with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
    fc.to_excel(writer, sheet_name='Pronóstico', index=False, startrow=0)
    wb = writer.book
    ws = writer.sheets['Pronóstico']

    # Gráfico nativo de Excel
    chart = wb.add_chart({'type':'line'})
    rows = len(fc)
    chart.add_series({
        'name':'Pronóstico',
        'categories':['Pronóstico', 1, 0, rows, 0],
        'values':['Pronóstico', 1, 1, rows, 1],
        'marker':{'type':'circle','size':4},
    })
    chart.set_title({'name':f'Pronóstico ({model_name}) a {h} meses'})
    chart.set_x_axis({'name':'Fecha'})
    chart.set_y_axis({'name':'Ventas'})
    ws.insert_chart('D2', chart, {'x_scale':1.2, 'y_scale':1.2})

    # Información adicional
    start = rows + 3
    info = [
        "Informe: Agente de Pronósticos – Nazdi IA",
        f"Fecha: {pd.Timestamp.now()}",
        f"Modelo: {model_name}",
        f"Horizonte: {h} meses"
    ]
    for i, line in enumerate(info):
        ws.write(start + i, 0, line)

# Ya no es necesario writer.save()
output.seek(0)

# 7) Botón de descarga
st.download_button(
    "⬇️ Descargar Excel completo",
    data=output,
    file_name="pronostico_agente.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
