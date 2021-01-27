---
permalink: /concurso/
layout: page
title: "Desafío Itaú-Binnario"
subheadline: ""
teaser: "Lecciones aprendidas luego de mi primera Competencia."
---
<!-- ...and learn more at the same time. You can message me at the [contact page]({{ site.baseurl }}/contact/). -->

{% comment %} Including the two pictures is a hack, because there's no way use the Foundation Grid (needs a container with row) {% endcomment %}
![picture of me]({{ site.urlimg }}widget1-trofeo-binnario.png){: .left .show-for-large-up .hide-for-print width="500"}
![picture of me]({{ site.urlimg }}widget1-trofeo-binnario.png){: .center .hide-for-large-up width="500"}
Para los que no saben, esta es mi primera competencia de Machine Learning. La verdad siempre quise participar, pero no me atrevía, le tenía susto a no dar el ancho, y a pesar de que soy bastante seguro de lo que sé, uno nunca piensa que puede ganar cuando están compitiendo los mejores.

En Kaggle existe la *"tradición"* que al finalizar la competencia se comparte la solución ganadora (al menos lo que se puede) por parte de los ganadores. Y dado que uno de mis objetivos es desarrollar contenido en Ciencia de Datos en Español y porque tengo un espíritu docente, pretendo contar cómo fue el ganar y qué cosas es necesario poner atención para que en la comunidad en Chile **vayamos aprendiendo**. Mi intención es explicar cómo fue el proceso de competencia y tratando de dar crédito a todos aquellos que fueron parte de este proceso y que no pude agradecer [acá](https://www.linkedin.com/feed/update/urn:li:activity:6754831905763471360/).

{% include alert warning='Perdón si doy detalles muy escuetos del modelo, es que, no puedo contar cuál fue la solución ganadora. Pero trataré de contar qué lecciones aprendí.'%}


Esto partió un día de clases en la Academia Desafío Latam, y mi ayudante [Tamara Zapata](https://www.linkedin.com/in/tamarazapatag/), me dice <q>¿viste el concurso, deberíay participar, demás ganay?</q> (Demasiado optimista para mi gusto). Nunca lo dije, pero me bajó el susto, ni loco se puede ganar esto, si hay tanta gente inscrita, creo que al final fueron alrededor de 1000 equipos, y mucha gente que sabe. Aparte el premio, USD$20.000, era muy llamativo por lo que era muy probable que muchas personas quisieran participar. Más detalles del concurso [acá](https://binnario.ai/challenge/-MMDsMov6MVyOl3gDuOB).

Ese día después de la clase nos quedamos revisando la data, y lo primero que recuerdo fue: <q>no tengo idea de la métrica</q>. Googleando, ví que era una métrica bastante famosa en *object detection* y en modelos de recomendación, por lo que primero pensé fue, quizás podría probar algunos modelos de recomendación en particular <mark>Factorization Machines</mark> que era algo que había estado leyendo últimamente.

{% include alert info='Factorization Machines son un tipo de Modelos no muy conocidos y no tan populares al menos en Chile, que permiten utilizar descomposición en factores latentes, normalmente factorizando matrices para generar modelos de Filtrado Colaborativo.'%}

Luego notamos que los datos no tenían estructura de modelamiento, eran datos tal cual salían de una base de datos. Por lo tanto había un paso previo. Y eso fue.

> A todo esto, es primera vez que ganó algo importante de este calibre, me he destacado, pero por ahí entre el montón, entonces obviamente esto es un tremendo logro, pero más que nada tremenda sorpresa!!

Bueno, días después me dió por probar. Me puse a hacer **EDA**. Revisé la cantidad de datos, las distintas tablas, igual era un problema grande, entre train y test eran fácil 35 millones de registros contando todas las tablas. Y empecé a sacar conclusiones de la data que había y de lo que no había. Entonces dije voy a hacer un modelo rápido, me gusta utilizar Random Forest como modelo Baseline, así que rápidamente entrené un modelo y quedé en `3er lugar`. 

> Claramente no me lo esperaba, y ahí dije: *"ya, quizás sé como resolver este problema"*, y si bien la métrica dio bien baja, estaba tercero. Al ocurrir eso, pensé bueno tengo posibilidades de estar dentro de los 10 primeros lugares y en verdad, con presentar frente a Itaú me conformaba.

Ahí hubo un cambio, porque empecé a destinar varias horas en la noche para modelar. El tema es que entremedio de este proceso me encontraba terminando de escribir mi Memoria, por lo que me encontraba agotado entre mi trabajo, las clases en la Academia, la Memoria y ahora esto. Así que decidí pedir vacaciones para poder relajarme, pasar tiempo con mi familia, pero también dedicarme más de lleno a esto.

Esto me permitió poner full atención al modelo en las noches, que es mi peak de concentración. Salieron varias ideas, y al cambiar a modelos más potentes empecé a mejorar. Ya en mi 3era submisión quedé en el 1er lugar, y superando la barrera sicológica del 50%, que a ese nivel de la competencia estaba muuuuy dura (creo que no más de 3 personas habían pasado el 50%). Ya ahí me dí cuenta de que habían reales posibilidades y me lo tomé en serio. Pensé en utilizar mi Laptop en su máxima capacidad aprovechando que me había comprado un Legion 5 nuevito de paquete.


## El primer problema, 1era Lección: Prepara tu Compu

Me quedé corto de RAM, y varios modelos ya no podían correr, estaba creando muchas variables y mi PC no daba (a ese tiempo tenía 16GB de RAM). Así que tomé la decisión de jugármela, dije <q>hay que sacar al menos un podio sí o sí para pagar esta RAM, y subí a 32GB</q>.

El tema es que al cambiar las RAM nos dimos cuenta que mi compu tenía un slot de RAM quemado y no podía aumentar. Tuve que viajar a Santiago en `cuarentena`, fue redificil, para cobrar la garantía. Lamentablemente no habían más laptops de esas características, y no tenía cómo solucionarlo. Me ofrecieron la devolución del dinero y en modo urgente tuve que ir al *Parque Arauco* a llevarme un computador `ya`. 

No estaba el Legion 5, y un <mark>tremendo vendedor</mark>, (se notaba que sabía) me metió un Legion 7 y me *"engrupió"* con que traía una RTX 2070. Y no me pude resistir, obviamente este Laptop salió harto más caro que el Legión 5, entonces ahora sí necesitaba sacar un podio para pagarlo. El tema es que el compu no era con retiro inmediato si no que se envíaba, y bueno saben la reputación de los envíos no es la mejor por la pandemia, no siempre llegan cuando dicen, pero este cabro me *"recontrajuró"* que en 4 días llegaba. Así que me la jugué, aproveché esos días para terminar mi memoria en un compu rasquita que tenía y afortunadamente el Legion 7 llegó en 3 días. Reinstalé todo, y entre la falla de mi Legion 5 hasta la puesta en marcha del Legion 7 pasaron 10 días sin poder modelar nada. Mantuve el primer lugar, lo cual me tenía preocupado, pero apenas me puse a probar nuevos modelos aparecieron los pesos pesados, casi todos los finalistas empezaron a superar el 50% y me tiraron muy atrás en el leaderboard.

## Otro Problema: 2da Lección

![picture of me]({{ site.urlimg }}concurso/codigo.jpeg){: .left .show-for-large-up .hide-for-print width="500"}
![picture of me]({{ site.urlimg }}concurso/codigo.jpeg){: .center .hide-for-large-up width="500"}
Un segundo problema, ya casi en la recta final del concurso fue, bueno, los modelos que generé fueron casi todos a la rápida, ¿cómo me ordeno? Y aquí viene la otra lección: <mark>MODULARIZACIÓN</mark>.

Comencé a Modularizar mi código y definir distintas partes lo más automatizadas posibles, normalmente tenía una parte bien al lote pero que me dio mucho valor que fue el análisis exploratorio, pero luego todo como relojito:

* **Generación de Variables**: Me preocupé de encapsular todos las variables nuevas que requirieran gran proceso de cálculo en funciones sumamente flexibles, si necesitaba generar más, sólo variaba parámetros y eso me permitía gran poder de experimentación, generando muchos cambios con sólo modificar algunos parámetros.

* **Código de Entrenamiento**: También me preocupé de modularizar esto, tratar de encapsular el preprocesamiento fue complejo, porque habían muchos tipos distintos dependiendo del modelo, pero luego los modelos que fueron entregando los mejores resultados se empezaron a acotar y eso me permitió simplificar el Script.

* **Inferencia**: Este notebook se encargaba de poder generar las predicciones pero además generar una estrategia de ordenamiento de las predicciones que era parte del problema. Finalmente se transforma la predicción final en el formato requerido por la plataforma para la predicción final.

{% include alert success='Acá no descubrí nada nuevo, luego siguiendo a uno de mis Youtubers favoritos hoy, Abishek Thakur, y leyendo su [libro](https://github.com/abhishekkrthakur/approachingalmost/blob/master/AAAMLP.pdf), me dí cuenta que el propone un sistema muy similar, pero muchísimo más robusto que el mío.'%}

## Otra Lección: Aprender del error

Luego de muchos experimentos, noté que no estaba llevando una manera óptima los experimentos. Lo peor de todo es que tengo un tutorial de MLflow, y <mark>NO LO UTILICÉ</mark>. Obviamente esto me llevó a notar que hay muchas herramientas para trackear experimentos, y hay una particular que me está cautivando harto y es [Weigths & Biases](https://wandb.ai/).

Definitivamente noté un gran valor en llevar los resultados del mejor modelo, y eso lo hice, pero también después noté que hay un gran valor en llevar el registro de los experimentos no tan exitosos, porque me pasó que como probé muchas cosas, se me olvidaba que había probado. Eso me generó mucha inseguridad porque llegó un momento en la competencia que los puntajes se estancaron, en alrededor de un 55% yo creo que iba 3ro y no tenía muchas más ideas. Y no sabía cuáles retomar y cuáles no.

{% include alert alert='Una dificultad que tenía esta competencia es que sólo indicaba el puntaje si éste era mejor que el anterior, en caso de no serlo, uno nunca sabía si había errado por poquito o estaba muy perdido.'%}

## Búsqueda de Hiperparámetros

![picture of me](https://optuna.org/assets/img/blog-multivariate-tpe.gif){: .right .show-for-large-up .hide-for-print width="500"}
![picture of me](https://optuna.org/assets/img/blog-multivariate-tpe.gif){: .center .hide-for-large-up width="500"}
Una de las cosas que para mí tiene mucho valor es el tema de entender qué hacen los hiperparámetros en un modelo. En general leo mucho en la documentación acerca de qué significa, y tiendo a depender mucho de `GridSearch`. Voy mezclando esto con tuneo manual, para sacarle el jugo al modelo.

Esto tiene sus ventajas, tengo control total sobre la búsqueda de hiperparámetros, pero, depende de que yo esté atento y además GridSearchCV en Scikit-Learn es muy ineficiente y tiende a ser lento y explotar mucho debido a su consumo de RAM.

Esto me ayudó mucho a poner en práctica sobre algunas cosas que aprendí hace tiempo, y que nunca había tenido que aplicar: `Downcasting` y el uso de `Matrices Sparse`, que en mi caso aplicaba. Esto me ayudó mucho a reducir el tamaño del dataset de train y permitir que el GridSearch funcionara, pero... hay manera más eficientes.

Para el futuro, pretendo utilizar un framework de Cross Validation más estático, al final es el KFold lo que aumenta mucho el costo de evaluación del modelo, por lo tanto, pretendo aplicar estrategias como las de [Abishek](https://www.youtube.com/watch?v=ArygUBY0QXw).



Además, nunca me había dado el tiempo de aprender librerías específicas de optimización de Hiperparámetros, especialmente porque intenté aprender hyper-opt una vez y la encontré muy engorrosa y nunca encontré necesario usarla. Obviamente, modelos grandes requieren una estrategia más Bayesiana para buscar de manera más inteligente y encontré que `Optuna` es una tremenda herramienta y la estaré agregando a mi arsenal.

## Soft Skills

![picture of me]({{ site.urlimg }}concurso/present.png){: .right .show-for-large-up .hide-for-print width="500"}
![picture of me]({{ site.urlimg }}concurso/present.png){: .center .hide-for-large-up width="500"}
Llegando ya a la última semana aparecieron los pesos pesados, y yo estaba sin ideas. [Joaquín](https://www.linkedin.com/in/joaqunv/) y [Malen](https://www.linkedin.com/in/malen-antillanca/) cada vez que subían un modelo me pasaban y se instauraron rápidamente en el primer lugar, y apareció tambíen los equipos de Rodrigo y [Paul](https://www.linkedin.com/in/paul-bertens-ai/) que hicieron pocas submisiones pero subían muy rápido. El tema es que decidí ocupar todo lo que se me ocurría en el momento y el día Jueves antes de finalizar la competencia quedé en 2do lugar detrás de Joaquín y Malen, y dije: <q>estoy cansado, esto es todo, con tal de presentar estoy contento/</q>. Al levantarme el viernes ya iba 5to, y al final del día quedé 8vo que fue mi puesto final, y podría haber quedado fuera. 

{% include alert tip='Acá hay que ser muy estratégico, porque no supe como generar mis últimas submisiones, de hecho Paul y Rodrigo envíaron su submisión final a las 23:51 pasando en último momento a Joaquín y Malen. Entonces realmente hay que ser estratégico y pensar bien cuando quemar todos tus cartuchos.' %}

Para rematarme, ese día salió un reportaje en [`LUN`](https://www.lun.com/Pages/NewsDetail.aspx?dt=2020-12-11&PaginaId=24&bodyid=0) con los ganadores del año pasado, los fundadores de Acacia, Catalina Espinoza, y Abelino Jiménez. Ella, Magíster de la Universidad de Chile, él, Doctor en Carnegie Mellon, yo... un egresado de Ingeniería Civil, hablaron de su solución con Redes Neuronales y dije... <q>uuuhhhh, en el Pitch no tengo oportunidad... a menos, que haga mi modelo interesante</q>. Y obvio, no todas las competencias se ganan así, pero esta sí. La habilidad de comunicar es tan importante como la habilidad para modelar. Creo que la competencia dejó en evidencia que varias personas saben hacer la pega... para más remate, Catalina y Abelino estaban entre los 3 finalistas y habían Estudiantes de Doctorado en AI, Profesores de Magíster, gente de Argentina, Perú, etc. ¿Cómo ganarles? Bueno, confiando en tu trabajo, entiendo fortalezas y debilidades de tu modelo y haciéndolo relevante. Para ello varios consejos:

* **Entendiendo el problema de Negocio**: Intuyendo para qué se usan estos modelos y poniendo atención a respuestas de los organizadores. Revisar comentarios en el Slack de la competencia me ayudó a entender cómo se va a consumir el modelo, y cómo tu modelo suple sus necesidades.

* **Poniendo atención al Slack**: Mucha de las preguntas que otros hicieron me ayudaron a mejorar mi modelo y a preparar mi Pitch final, leí todo el Slack. Además, leyendo sutiles pistas de los organizadores ayudaron a confiar en que mi modelo tenía potencial, a pesar de quedar 8vo enfocándome en cómo mi modelo resolvía problemas reales que ellos tenían.

![picture of me]({{ site.urlimg }}concurso/modelo.png){: .left .show-for-large-up .hide-for-print width="500"}
![picture of me]({{ site.urlimg }}concurso/modelo.png){: .center .hide-for-large-up width="500"}

* **Preparar el Pitch**: Son 4 minutos en los que te juegas la vida, hicimos 3... y es la peor experiencia por la que ha pasado. El corazón a mil y la verdad es que da mucho nervio. Si te trabas, no alcanzas en el tiempo y en verdad fueron super rigurosos con eso. Traté de incluir comentarios que se hicieron en Slack o durante la conversación de finalistas tratando de demostrar siempre como mi modelo suplía sus necesidades. 

La preparación de esos 4 minutos de Pitch me tomaba fácil 6 horas, y mucha práctica, realmente había que escoger muy bien qué decir y qué no decir. Para el último Pitch estaba muy cansado y la verdad es que el día anterior me fui a acostar sin tenerlo tan bien preparado. 

Pero bueno el día de la final desperté a las 7am, y estuve 3 horas practicando el Pitch final (a las 10am), pero no quedaba conforme, me trababa, se me olvidaban partes, no sé, entré a la presentación sin estar 100% convencido. 

Afortunadamente me fue bien, y cuando me avisaron que mi modelo era el ganador casi me da un ataque, pensé que me estaban bromeando. Fue bueno ver que el esfuerzo de años, aprendiendo por mi cuenta, no siendo considerado o sencillamente siendo mirado en menos por la falta de mi título al final daba frutos.

Hay harta gente que agradecer en el camino: Mi esposa `Valentina`, que me dio permiso para acostarme tarde por más de un mes, y ella es muy friolenta. Pero desde la `Tam` que me avisó de la competencia, o amigos como Lautaro, yo creo que ni cacha, pero él me pasó el primer Libro de Machine Learning para leer, el [Applied Predictive Modeling](http://appliedpredictivemodeling.com/) de Max Kuhn, o amigos como Nico, que me hablaron de Deep Learning por primera vez, o Matías (el Dragon Slayer) que siempre me cuestionaba y que tenía que usar Python y me presentó técnologías como Github o Pytorch. **Gracias!**

![picture of me]({{ site.urlimg }}concurso/LogoBinarioBlanco.png){: .right .show-for-large-up .hide-for-print width="500"}
![picture of me]({{ site.urlimg }}concurso/LogoBinarioBlanco.png){: .center .hide-for-large-up width="500"}

Pero bueno, espero que esto pueda ser de utilidad para entender que aparte del modelo hay muchos aspectos a los cuales poner atención, y ojalá cuando haya una nueva competencia de [Binnario](https://binnario.ai/), todos podamos ser mejores y sigan con esta idea de aconsejar al resto.

[**Alfonso**]({{ site.baseurl }}/contact/)





