---
permalink: /datathon/
layout: page
title: "Attention is not always all you need!!"
subheadline: "Mi experiencia ganando la Datathon USM 2022."
teaser: "¿Qué hacer cuando los Transformers no dan resultado?"
---
<!-- ...and learn more at the same time. You can message me at the [contact page]({{ site.baseurl }}/contact/). -->

{% comment %} Including the two pictures is a hack, because there's no way use the Foundation Grid (needs a container with row) {% endcomment %}
![picture of me]({{ site.urlimg }}datathon.png){: .left .show-for-large-up .hide-for-print width="500"}
![picture of me]({{ site.urlimg }}datathon.png){: .center .hide-for-large-up width="500"}

## TL;DR

Y gané una segunda competencia. No saben la sorpresa que me llevé al enterarme de que obtuve un primer lugar de nuevo. Esta vez en una competencia completamente distinta, NLP. Bueno, les quiero contar qué significó este nuevo logro para mí, totalmente inesperado en un área en el que en verdad no soy muy experto.

Primero que todo, ¿Qué es la Datathon USM 2022? Bueno una competencia abierta a estudiantes de Pregrado y Postgrado para estudiantes en universidades Chilenas. El objetivo de la competencia era determinar Odio y además si es que se hace referencia a comunidades vulnerables, en este caso las comunidades a detectar corresponden a: Mujeres, Comunidad LGBTQ+, Comunidades Migrantes y Pueblos Originarios. Para ello se nos hizo entrega de un dataset de cerca de 5000 Tweets: 2200 aproximadamente para entrenar y 2900 para envíar al portal de la competencia para evaluación, los cuales no estaban etiquetados. Adicionalmente el puntaje en el Leaderboard no era el único aspecto importante. En este caso los 3 mejores puntajes tenían que presentar frente a un grupo de jueces los cuales dictaminarían el podio final. Creo que el gran atractivo de esta competencia es que los Tweets estaban en su gran mayoría en Español Chileno, es decir, contaban con mucho doble sentido, modismos muy chilenos, y mucha pero mucha grosería asociada al Chileno. De hecho habían tweets muy pero muy ordinarios.

{% include alert alert='Se van a encontrar con varios Tweets de ejemplos a lo largo del artículo que tienen lenguaje muy ofensivo. Pido perdón de antemano, porque precisamente uno de los objetivos lindos de la competencia es chequear si nuestros modelos pueden detectar este tipo de lenguaje para poder combatir este tipo de comportamiento totalmente indeseable.' %}

Bueno, nuevamente, me enteré de la competencia por un tercero. Uno de mis profesores guías (y Coordinador del MSDS), [Sebastian Moreno](https://www.linkedin.com/in/sebastian-moreno-araya/) envió un comunicado a todos los estudiantes del Magíster precisamente invitando a participar en esta competencia. Si bien el premio en dinero, no era tan sabroso como en el [Desafío Itaú-Binario]({{ site.baseurl }}/concurso/), creo que el problema era muchísimo más interesante. Me llamó principalmente la atención porque era la oportunidad de poner en práctica uno de los temas que me tiene muy entusiasmado, los famosos Transformers. 

Los transformers solía ser el gran tema del Deep Learning (todavía lo son, pero Stable Diffusion está muy en boga por tantos modelos generativos que andan dando vuelta por ahí). Obviamente hace mucho tiempo que quería poner esto en práctica y de hecho quería probar un modelo que hace mucho tiempo me tenía intrigado, el [Beto](https://github.com/dccuchile/beto). Beto es un modelo Bert preentrenado en lenguaje español por la Universidad de Chile. Es un tremendo modelo que ha dado muy buenos resutados en distintas tareas de lenguaje Natural y claramente era la oportunidad de poner en prácticas varias avanzadas de Deep Learning. Pero antes de entrar en los detalles de la solución vamos al problema a resolver. 

## El Problema

Como ya mecioné, el problema contaba con 2256 Tweets de entrenamiento, los cuales contenían 9 columnas: un `tweet_id`, `author_id`, `conversation_id`, `text`, y 5 clases asociadas a si hasta 3 anotadores determinaban la presencia de Odio, y/o referencias hacia las comunidades de Mujeres, LGBTQ+, Migrantes o Pueblos Originarios. Creo que el primer punto a destacar acá es que este dataset fue construido especialmente para la competencia y es fruto del trabajo de tesis de Domingo Benoit (alumno de Pregrado de Informática) y varios colaboradores, a los cuales me encantaría poder dar el crédito que corresponde. 

El problema estaba diseñado como una clasificación Multilabel. Es decir, cada clase podía tomar un valor 0 o 1 de manera independiente. Al menos el test set se evaluaría de esa manera. Y bueno la primera complejidad del problema es que Tweet no tenía valores 0 o 1, sino de 0 a 3, que correspondía al número de Anotadores que indicó una cierta clase.

![picture of me]({{ site.urlimg }}datathon/ejemplo_tweet.png){: .center}

Por ejemplo para un Tweet dado, se indicaba qué etiquetas tenían anotadores que indicaron alguna de las clases. Y si se dan cuenta, hay una componente bien subjetiva en la anotación. En el caso de la imagen el tweet mencionado es considerado Odio sólo por dos anotadores. Y sólo uno considera una alusión a la Comunidad LGTBQ+.

{% include alert info='Según se nos comentó, la anotación de los tweets fue realizada por 3 anotadores a ciegas, los cuales no tuvieron interacción entre ellos a modo de llegar a algún acuerdo en la anotación. Según se indicó fueron 2 hombres y una mujer.' %}

Como se puede ver esto, ya podría ser una fuente de sesgo, y la primera decisión importante a tomar era como generaríamos etiquetas para el entrenamiento del modelo. Con respecto a las otras variables como `tweet_id`, `author_id`, `conversation_id`, yo no las utilicé. `tweet_id`, `author_id` eran sólo identificadores, mientras que `conversation_id` permitía unir el tweet a una conversación, el cuál era un set complementario que se nos entregó. Supuestamente esto permitía que el modelo tuviera un contexto más amplio, pero en mi caso no le encontré utilidad debido a que:

* Muy pocos tweets tenían un hilo, de conversación asociado. 
* Existían tweets de contexto, que venían en idiomas distintos, por ejemplo, inglés. 
* Luego de inspeccionarlos de manera manual, pude notar que no entregaban información relevante para identificar alguna de las clases. 
* Adicionalmente, pensé que esta información se podía utilizar para aumentar el dataset de entrenamiento que era algo permitido en la competencia. Pero la verdad es que era tan complejo el criterio de etiquetado, que no era sencillo aumentar el dataset.

## Sobre la evaluación

El modelo se evaluaba en una métrica bien extraña a la que llamaron F1 Custom. Consistía en un F1 calculado como el 50% de un Binary F1 Score para la clase Odio, y un 50% del Macro F1 Score para la comunidades vulnerables. Esto inmediatamente nos indicaba que la clase más importante a predecir era el Odio, ya que ponderaba mucho más que cada una de las otras 4. Interesantemente luego de conversar con los organizadores en la premiación, me contaban que la razón de esta métrica fue más por limitaciones de la plataforma. Aún así terminó siendo una métrica bien interesante, y que costaba mucho subir.

## Mi problema

Yo la verdad entré a la competencia pensando que tendría tiempo suficiente para poder competir bien, pero lamentablemente las instancias finales me toparon con la peor parte de mi semestre. He estado distante del blog, y de Linkedin, y de la vida en general, porque la verdad es que el segundo semestre del MSDS está muy intenso. A eso, sumarle una competencia, no era una buena idea. Por lo tanto, tenía que utilizar cada segundo de mi tiempo de manera precisa para poder rendir en todas las cosas que estoy haciendo. Es por eso que una de las primeras cosas que hice, fue mirar las lecciones aprendidas del desafío [Itaú-Binario]({{ site.baseurl }}/concurso/) y en el [Tabular Playground]({{ site.baseurl }}/kaggle-tps/), tratando de no repetir los mismos errores.

## Planteamiento del problema

Primero que todo en este caso traté de utilizar buenas prácticas de código, y ser lo más ordenado posible. Me funcionó medianamente. El tener que iterar tan rápido, hace que cueste mucho programar de manera ordenada. Por lo que si bien traté que mis notebooks fueran ordenados, no pude seguir una lógica de scripts como sí hago en proyectos reales. La razón de esto es, principalmente, que para mí es más rápido debuggear en notebooks que en Scripts. 

Lo segundo, que fue una lección aprendida es siempre confiar en tu CV. Y en este caso, esto fue lo primero que hice. Definí que mi esquema de validación iba a ser K-Fold. Y una de las cosas novedosas de mi solución, para chequear que efectivamente mi validación era confiable, fue utilizar Adversarial Validation.

![picture of me]({{ site.urlimg }}/../../images/datathon/adversarial_val.png){: .center width="500"}

Adversarial validation es una metodología sumamente simple introducida por Uber el 2020 para poder determinar si hay Concept Drift. El concept Drift es cuando la distribución de los datos cambia en el tiempo. En el caso de una competencia, nosotros necesitamos chequear si el Train y Test set don i.i.d (independent and identically distributed). Esto nos permite confiar de que la data usada para entrenar tendrá una distribución similar que la de test. Si esto ocurre, entonces nosotros podemos confiar en nuestro esquema de validación (que principalmente utiliza datos de train, porque el test no está etiquetado) es un buen proxy de la generalización de nuestro modelo. 

En simple Adversarial validation, entrena un modelo etiquetado como 1 si la observación proviene del train set y 0 si viene del test set (o puede ser al revés). Si luego de entrenar el modelo, métricas como el ROC AUC dan valores cercanos a 0.5, quiere decir que el modelo no es capaz de diferenciar de donde proviene la observación. Esto es un indicador que la distribución de train y test son muy parecidas/idénticas.

El poder contar con una validación de este tipo me permitió envíar sólo 12 submisiones en la competencia. Debido a que mi puntaje de CV Local estaba correlacionado con el Leaderboard (y esta garantía me la entregó este método). Eso me permitió experimentar muchísimo sin estar restringido a las 2 submisiones diarias permitidas en la competencia.

{% include alert warning='Al menos a  lo largo de la competencia mi esquema de validación, estuvo bastante bien, excepto cuando probé pseudolabeling. Al probar esta metodología, el leaderboard me mostró que sobreajuste. Obtuve mi mejor puntaje en mi CV Local, pero esto no se tradujo en mi mejor puntaje en el Leaderboard. Ahora la diferencia fue poca, esperaba un valor cercano al 0.82 y obtuve sólo un 0.81. Durante las presentaciones finales nos dijeron que el leaderboard evaluaba sobre un 70% de los datos de test. Pero ese 70% iba cambiando. Esto no es una práctica común en las competencias, pero probablemente afectó en nuestra experimentación, y nos impidió obtener aún mejores resultados.' %}

## Sobre la solución

Bueno, mi primer intento fue obvio, lo que está de moda y que venía como anillo al dedo a la competencia: Un transformer. En mi caso tenía dos opciones: Beto, un Bert preentrenado en datos en español por la Universidad de Chile. De hecho hace mucho tiempo que quería probar este modelo, y de hecho los otros dos equipos: LonelyWolf y MoccaOverflow lo utilizaron. Pero obvio, no lo probé, porque me incliné por [Robertuito](https://arxiv.org/abs/2111.09453). Todo indicaba que un transformer como este era el más indicado:

* RoBERTa: Se supone que una versión más robusta que Bert. Es el mismo modelo pero entrenado en más datos de una manera más robusta.
* Entrenado en Tweets en Español: Robertuito es un modelo pre-entrenado en 500 millones de Tweets en español. Además el modelo viene equipado con un preprocesamiento especial para Tweets, en el cual se genera un tratamiento a los handles (@), hashtags (#), URLs y emojis. Esto lo encontré particularmente interesante, porque permitía transformar los emojis en descripciones en lenguaje natural, que en este caso podría haber sido de mucho utilidad.

![picture of me]({{ site.urlimg }}datathon/ejemplos_preprocess.png){: .center}

* Robertuito fue entrenado en varias tareas, una de ellas, Hate Speech Recognition. Que resultaba ser la tarea en cuestión. 

Yo de inmediato pensé que con esto, el problema estaba resuelto. De hecho voy a publicar este modelo, porque su implementación es bien interesante. Pero lamentablemente no funcionó. O al menos no funcionó a nivel de competencia.

El puntaje de CV local fue bien variado, desde 0.49 para la clasificación Multilabel (recordar que Robertuito estaba pre-entrenado en detección de Odio, pero no de Comunidades) hasta 0.73 cuando le apliqué Fine-Tuning en sólo la clase Odio.

<q>Entonces, ¿Por qué un modelo tan sofisticado como un Transformer no funcionó?</q> 

Tengo varias teorías.

1. El Robertuito tenía sobre 84 Millones de Parámetros, y nosotros teníamos poco más de 2000 tweets con etiquetas para hacer fine-tuning. Probablemente no le hicimos ni cosquillas a los parámetros, ya que era muy poca data.
2. La otra razón, yo creo que tiene que ver netamente con la tokenización. Por favor muéstrenme cómo se tokeniza algo así:

![picture of me]({{ site.urlimg }}datathon/tokens.png){: .center}

Si son Chilenos, sabrán que esto terriblemente grosero. Y utiliza una combinación de garabatos/groserías, modismos, y palabras muy propias del chileno, que creo que es imposible para un modelo pre-entrenado entender. Principalmente porque los modelos pre-entrenados usan datos de español global. Y realmente el Chileno es casi un idioma en sí.

De hecho, al ponerme a buscar en internet al respecto, me encontré con un [artículo](https://www.elmundo.es/cultura/2021/11/30/61a4a36321efa013518b4571.html) del Diario el Mundo, que precisamente hacía alusión a este tipo de "problemas" del Chileno.

Entonces, ¿cómo afrontar un problema donde tu mejor arma, no da resultado? Podría haber intentado con el Beto, como los hicieron los otros chicos, pero un poco tenía miedo de que terminar con los pobres resultados de Robertuito. Entonces, cuando lo complejo no funciona, hay que volver a lo simple. Y aquí es donde decidí implementar un clásico Bag of Words. 

{% include alert todo='Debido a la competencia, seguí investigando sobre el tema, y la verdad es a pesar de que los transformers la llevan cuando se trata de NLP, el Bag of Words (BOW) todavía tiene algo que decir. En especial en formas textuales que no cuentan con el nivel de investigación que tiene el inglés vale la pena. Es más, existen formas de extraer, palabras, frases y frases claves utilizando transformers para luego alimentar modelos más simples de BOW.' %}

Bueno, yo sabía que especialmente habiendo una fase de presentación frente a jueces, un Bag of Words podría ser un arma de doble filo. Obviamente es un modelo que puede ser demasiado sencillo, pero que increíblemente entregó buenos resultados, quizás por cómo funciona el chileno. En especial, el odio, se caracteriza por su alto nivel de garabatos, y palabras ofensivas. Al igual que como ocurre con el SPAM, pensé que quizás bastaba con encontrar palabras claves, y que eso sería suficiente para determinar odio y las comunidades vulnerables.




[**Alfonso**]({{ site.baseurl }}/contact/)





