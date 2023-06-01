# Pronóstico del precio de criptomoneda BTC-USD

![btc-usd](https://s2.coinmarketcap.com/static/img/coins/200x200/2083.png)

El objetivo de este proyecto es emplear una red neuronal para pronosticar el precio de la criptomoneda Bitcoin, tomando como referencia los datos históricos del precio del mes de abril de 2023, esto nos permitirá  comparar los resultados con el costo real del mes de mayo 2023.  

## La metodología planteada para este proyecto está estructurada de la siguiente forma
- Recopilación de datos: Se planea utilizar Yahoo Finance, para la recolección de datos de los precios de cierre diarios del  mes de abril de 2023.
- Preparación de los datos: Los datos se organizarán y limpiarán, para asegurar que su formato es el adecuado para el entrenamiento de la red neuronal. Estos datos serán divididos en conjuntos de entrenamiento y de prueba.
- Configuración del entorno de desarrollo: La red neuronal será desarrollada en Python empleando las bibliotecas usuales para el desarrollo de redes neuronales,  TensorFlow y Keras.
- Diseño y entrenamiento del modelo: La red neuronal que se diseñará será de tipo feedforward. Los datos de entrada serán los precios pasados de la criptomoneda y la variable objetivo corresponderá a su precio futuro. El modelo será entrenado utilizando el conjunto de datos designado como “datos de entrenamiento”, en el proceso probablemente deberán ajustarse los hiperparámetros como el número de capas y neuronas, la función de activación y la tasa de aprendizaje.
- Evaluación del modelo: El rendimiento del modelo será evaluado utilizando el conjunto de datos de prueba. Empleando el error medio absoluto o el error cuadrático medio se califica la eficiencia del ajuste a los datos de prueba.
- Predicción del precio: El modelo entrenado será capaz de hacer predicciones del precio futuro del Bitcoin.
- Análisis de resultados: Las predicciones generadas se analizarán y se evaluarán por el modelo en función de los precios reales de la acción. Podríamos examinar la precisión de las predicciones y buscar patrones o tendencias en los resultados.
