---
title: "Time Series Classification"
subheadline: "Resolviendo problemas no tradicionales en Kaggle."
teaser: "LSTM Multivariadas + Pytorch Lightning."
# layout: page-fullwidth
usemathjax: true
category: quick
header: no
image:
    thumb: secrets/secret.jpg
tags:
- python
- tutorial
published: false
---


![picture of me]({{ site.urlimg }}secrets/secret.jpg){: .left .show-for-large-up .hide-for-print width="300"}
![picture of me]({{ site.urlimg }}secrets/secret.jpg){: .center .hide-for-large-up width="250"}
Cuando uno comienza a revisar tutoriales de redes neuronales es común encontrarse con los mismos problemas: Regresión o Clasificación Binaria (o en el mejor de los casos multiclase). Pero existen varios otros tipos de problemas que normalmente no se ven y que son sumamente aplicables. En este caso vamos a resolver un problema de Clasificación de Series de Tiempo. Esto significa que le entregaremos a nuestra red una o varias series de tiempo, y la red la clasificará. <!--more-->

Existen distintas aplicaciones de este tipo de problemas, en este caso lo que haremos es tomar información proveniente de los sensores de un robot para poder predecir en qué tipo de suelo se está moviendo. En el xxxx de Kaggle, se ha recopilado información de sensores de orientación, acelerómetros y velocidad angular. Cada una de estas variables será una serie de tiempo. Estas mediciones se han aplicado a 9 tipos distintos de suelo. La tarea es poder utilizar esta información para identificar el suelo en el que actualmente se está moviendo el robot.

Esto puede tener muchas aplicaciones prácticas, algunas que se me vienen a la mente por el rubro en el que he estado trabajando últimamente es poder determinar el estado de los pavimentos de carreteras, o determinar posibles salidas de ruta, entre  otros.

Debido a que este problema tiene como objetivo trabajar con data secuencial es que se hace natural el uso de Redes Recurrentes, en particular de las LSTM (Long Short Term Memory). 

Las redes recurrentes son....

Las LSTM son una extensión muchísimo más sofisticada de las RNN que permiten recordar secuencias mucho más largas. 





[**Alfonso**]({{ site.baseurl }}/contact/)
