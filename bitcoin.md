# Pronóstico del precio de criptomoneda BTC-USD

![btc-usd](https://s2.coinmarketcap.com/static/img/coins/200x200/2083.png)

El objetivo de este proyecto es emplear una red neuronal para pronosticar el precio de la criptomoneda Bitcoin, tomando como referencia los datos históricos del precio del mes de abril de 2023, esto nos permitirá  comparar los resultados con el costo real del mes de mayo 2023.  
- Detección de fraudes financieros
- Valoración de riesgos crediticios
- Optimización de carteras de inversión
- Análisis de sentimiento del mercado
- Pronóstico de precios de acciones

## La metodología planteada para este proyecto está estructurada de la siguiente forma
- Recopilación de datos: Se utilizaron los datos proporcionados por <b>Yahoo Finance</b>, correspondientes a los precios de cierre diarios del Bitcoin de los últimos 365 días. 
- Preparación de los datos: Los datos se organizaron, limpiaron y procesaron para asegurar que su formato fuera el adecuado para el entrenamiento de la red neuronal. 
- Configuración del entorno de desarrollo: La red neuronal fue desarrollada en <b>Python</b> empleando bibliotecas comunes para el desarrollo de redes neuronales,  <b>TensorFlow</b> y <b>Keras</b>.
- Diseño: La red neuronal que se diseñó fue de tipo <b>feedforward</b>. Se especificó el número de capas ocultas, el número de neuronas por capa y la función de activación.  
- Evaluación del modelo: El rendimiento del modelo fue gestionado a través del optimizador de funciones Adam y la función de pérdida seleccionada fue el error cuadrático medio. Una vez que el modelo recibe los datos se busca minimizar esta función empleando el optimizador,  con el propósito de ajustar los pesos de la red para que las predicciones se acerquen lo más posible a los valores reales.
- Entrenamiento del modelo: Aquí se realiza el proceso de aprendizaje mediante la optimización de los pesos de las conexiones neuronales. El modelo recibió  los datos guardados como (X) y aquellos guardados como (y), y el optimizador trabajó  sobre la función de pérdida para lograr el mejor ajuste. En este paso se seleccionaron el número de épocas y el tamaño del lote. 
- Predicción del precio: El modelo entrenado será capaz de hacer predicciones del precio futuro del Bitcoin.
Análisis de resultados: Examinamos la precisión de las predicciones, comparando con los datos reales. 

## Implementación
Primeramente se importan las librerías que serán utilizadas, como 'numpy', 'pandas' 'sklearn' y 'keras' de 'tensorflow'.
```python 
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
```

Se cargan los datos de precios de acciones desde el archivo CSV.
```python 
data = pd.read_csv("BTC-USD.csv")  
```
Se extraen los precios de la columna 'Close' correspondiente al DataFrame previamente cargado.
```python 
prices = data['Close'].values 
```
Se convierte la lista de precios en un arreglo de <b>numpy</b>.
```python 
prices = np.array(prices)
```
Se cambia la forma del arreglo de los precios para que tengan una dimensión adicional.<br/>
Ésto es necesario para el procesamiento de los datos y el entrenamiento del modelo.
```python 
prices = prices.reshape(-1, 1)
```
Se realiza el preprocesamiento de los datos utilizando 'MinMaxScaler' de 'scikit-learn'.<br/>
Se crea un objeto 'scaler' de 'MinMaxScaler' y se utiliza el método 'fit_transform() para escalar los precios en el rango de 0 a 1.
```python 
scaler = MinMaxScaler()
scaled_prices = scaler.fit_transform(prices)
```
Se dividen los datos en características (X) y etiquetas (y).<br/>
La característica X son los precios escalados hasta el penúltimo elemento y las etiquetas son los precios escalados desde el segundo elemento hasta el último.
```python 
X = scaled_prices[:-1]
y = scaled_prices[1:]
```
Se crea un modelo de red neuronal utilizando 'Sequential' de Keras.<br/>
El modelo tiene una capa oculta densa con 64 neuronas, función de activación ReLu y una capa de salida densa con una neurona.
```python 
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(1,)),
    keras.layers.Dense(1)
])
```
Se compila el modelo especificando el optimizador y la función de pérdida.<br/>
En este caso se utiliza el optimizador Adam y la pérdida de error cuadrático medio.
```python 
model.compile(optimizer='adam', loss='mean_squared_error')
```
Se entrena el modelo utilizando los datos de características (X) y etiquetas (y).<br/>
El modelo se entrena durante 100 épocas con un tamaño de lote de 32.
```python 
model.fit(X, y, epochs=150, batch_size=64)
```
Epoch 1/150
6/6 [==============================] \- 1s 5ms/step \- loss: 0.0885<br/>
Epoch 2/150
6/6 [==============================] \- 0s 4ms/step \- loss: 0.0558<br/>
Epoch 3/150
6/6 [==============================] \- 0s 3ms/step \- loss: 0.0332<br/>
Epoch 4/150
6/6 [==============================] \- 0s 4ms/step \- loss: 0.0188<br/>
Epoch 5/150
6/6 [==============================] \- 0s 3ms/step \- loss: 0.0117<br/>
\------------------------------------------------------------------<br/>
Epoch 149/150
6/6 [==============================] \- 0s 2ms/step \- loss: 0.0018<br/>
Epoch 150/150
6/6 [==============================] \- 0s 3ms/step \- loss: 0.0018<br/>
<keras.callbacks.History at 0x7ff2eb1bb880>
<br/>
<br/>
Se realiza predicciones utilizando el modelo entrenado.<br/>
Se pasan las características (x) al método 'predict()' y se obtienen las predicciones correspondientes.
```python 
predictions = model.predict(X)
```
En el bucle for se muestra una sección predicciones y precios reales.<br/>
Se invierten las transformaciones de escalar utilizando 'scaler.inverse_transform()' para obtener los precios predicho y reales en la escala original para imprimirse en pantalla.
```python 
for i in range(10):
    predicted_price = scaler.inverse_transform(predictions[i].reshape(-1, 1))
    actual_price = scaler.inverse_transform(y[i].reshape(-1, 1))
    print("Predicted Price:", predicted_price)
    print("Actual Price:", actual_price)
    print()
```
Predicted Price: [[29706.15]]
Actual Price: [[29906.662109]]

Predicted Price: [[29777.436]]
Actual Price: [[31370.671875]]

Predicted Price: [[31192.537]]
Actual Price: [[31155.478516]]

Predicted Price: [[30984.531]]
Actual Price: [[30214.355469]]

Predicted Price: [[30074.852]]
Actual Price: [[30111.998047]]

Predicted Price: [[29975.91]]
Actual Price: [[29083.804688]]

Predicted Price: [[28976.785]]
Actual Price: [[28360.810547]]

Predicted Price: [[28267.842]]
Actual Price: [[26762.648438]]

Predicted Price: [[26700.742]]
Actual Price: [[22487.388672]]

Predicted Price: [[22508.578]]
Actual Price: [[22206.792969]]
<br/>
<br/>
```python 
# Obtener el último precio registrado
last_price = prices[-1]

# Escalar el último precio
scaled_last_price = scaler.transform([[last_price[-1]]])

# Realizar la predicción
predicted_scaled_price = model.predict(np.array([scaled_last_price]))

# Invertir la escala de la predicción
predicted_price = scaler.inverse_transform(predicted_scaled_price.reshape(-1, 1))

# Imprimir el precio predicho
print("Predicted Price for tomorrow:", predicted_price)
```
1/1 [==============================] - 0s 32ms/step
Predicted Price for tomorrow: [[27010.096]]
<br/>
## Conclusiones
El precio del Bitcoin puede ser volátil y difícil de predecir debido a que existen muchos factores que afectan el precio del Bitcoin. Algunos de ellos son la demanda, las condiciones económicas y el desarrollo económico. <br/>Sin embargo, el análisis de los datos históricos y las tendencias del mercado puede proporcionar información sobre los futuros movimientos de precios.
<br/>
## Links
[Proyecto en Colab](https://colab.research.google.com/drive/1SusBNKlOAbf4yxgQ8mQ0vP8SG38M80LF?usp=sharing#scrollTo=4LMZQ24LD3vd)
<br/>
[Presentación en Canva](https://www.canva.com/design/DAFlHgS3U9s/IsCoq3G4Th5qQPO3AHk80A/edit?utm_content=DAFlHgS3U9s&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

