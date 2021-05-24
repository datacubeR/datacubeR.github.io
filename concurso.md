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
Para los que no saben, esta fue mi primera competencia de Machine Learning, y probablemente la primera para muchos acá en Chile. La verdad siempre quise participar, pero no me atrevía, le tenía susto a no dar el ancho, y a pesar de que soy bastante seguro de lo que sé, uno nunca piensa que puede ganar cuando están compitiendo los mejores.

Finalmente en Abril del 2021 dimos por finalizado el proceso de implementación del modelo ganador de la competencia en el Banco Itaú. Fue un proceso largo pero que espero prontamente dé sus frutos. Este modelo permitirá priorizar el contacto del banco para con sus clientes de una forma que permita optimizar los recursos del banco, pero también contactar a clientes que efectivamente tengan una alta chance de adquirir el producto ofrecido, de manera de disminuir la tasa de molestia por clientes que no quieren ser contactados.

En Kaggle existe la *"tradición"* que al finalizar la competencia se comparte la solución ganadora (al menos lo que se puede) por parte de los ganadores, y dado que oficialmente se cerró la implementación del modelo y del pago del premio considero que sería un buen momento para comentar mi experiencia en la competencia. Además, uno de mis objetivos es desarrollar contenido en Ciencia de Datos en Español, por lo que espero que contar cómo fue la experiencia de participar/ganar sea de gran ayuda para que en próximas competencias todos estemos más preparados en cuanto a qué cosas es necesario poner atención. Mi intención es explicar cómo fue el proceso de competencia y tratando de dar crédito a todos aquellos que fueron parte de este proceso y que no pude agradecer [acá](https://www.linkedin.com/feed/update/urn:li:activity:6754831905763471360/).

{% include alert warning='Perdón si doy detalles muy escuetos del modelo, es que, no puedo contar cuál fue la solución ganadora. Pero sí trataré de contar qué lecciones aprendí.'%}


Esto partió un día de clases en la Academia Desafío Latam, y mi ayudante [Tamara Zapata](https://www.linkedin.com/in/tamarazapatag/), me dice <q>¿viste el concurso, deberíay participar, demás ganay?</q> (Demasiado optimista para mi gusto). Nunca le dije, pero cuando me dijo eso me bajó el susto y la presión de que había demasiada gente inscrita. Creo que al final fueron alrededor de 1000 equipos, y mucha gente que sabe. Aparte el premio de USD$20.000, era muy llamativo por lo que era muy probable que muchas personas quisieran participar. Más detalles del concurso [acá](https://binnario.ai/challenge/-MMDsMov6MVyOl3gDuOB).

Ese día después de la clase nos quedamos revisando la data, y lo primero que recuerdo fue: <q>no tengo idea de la métrica</q>. Googleando, ví que era una métrica bastante famosa en *object detection* y en modelos de recomendación, por lo que primero pensé fue, quizás podría probar algunos modelos de recomendación, en particular <mark>Factorization Machines</mark>, que era algo que había estado leyendo últimamente.

{% include alert info='Factorization Machines son un tipo de Modelos no muy conocidos y no tan populares al menos en Chile, que permiten utilizar descomposición en factores latentes, normalmente factorizando matrices para generar modelos de Filtrado Colaborativo.'%}

Luego notamos que los datos no tenían estructura de modelamiento, eran datos crudos, tal cual salían de una base de datos. Por lo tanto, había un paso previo de limpieza. Y eso fue todo lo que hicimos.

Bueno, días después me dió por probar. Me puse a hacer **EDA**. Revisé la cantidad de datos, las distintas tablas, igual era un problema grande, entre train y test eran fácil 35 millones de registros contando todas las tablas. Y empecé a sacar conclusiones de la data que había y de lo que no había. Entonces dije: <q>voy a hacer un modelo rápido!</q>. Me gusta utilizar Random Forest como modelo Baseline, así que rápidamente entrené un modelo y quedé en `3er lugar`. 

> Claramente no me lo esperaba, y ahí dije: *"ya, quizás sé como resolver este problema"*, y si bien la métrica dio bien baja, estaba tercero. Al ocurrir eso, pensé que quizás había posibilidades de estar dentro de los 10 primeros lugares y en verdad, con presentar frente a Itaú me conformaba.

Ahí hubo un cambio, porque empecé a destinar varias horas en la noche para modelar. El tema es que entremedio de este proceso me encontraba terminando de escribir mi Memoria, por lo que estaba bien cansado entre mi trabajo, las clases en la Academia, la Memoria y ahora esto. Así que decidí pedir vacaciones para poder relajarme, pasar tiempo con mi familia, pero también dedicarme más de lleno a esto.

Esto me permitió poner full atención al modelo en las noches, que es mi peak de concentración. Salieron varias ideas, y al cambiar a modelos más potentes empecé a mejorar. Ya en mi 3era submisión quedé en el 1er lugar, y superando la barrera sicológica del 50%, que a ese nivel de la competencia estaba muuuuy dura (creo que no más de 3 personas habían pasado el 50%). Ya ahí me dí cuenta de que habían reales posibilidades y me lo tomé en serio. Pensé en utilizar mi Laptop en su máxima capacidad aprovechando que me había comprado un Legion 5 nuevito de paquete.


## El primer problema, 1era Lección: Prepara tu Compu

Me quedé corto de RAM, y varios modelos ya no podían correr, estaba creando muchas variables y mi PC no daba (a ese tiempo tenía 16GB de RAM). Así que tomé la decisión de jugármela, dije <q>hay que sacar al menos un podio sí o sí para pagar esta RAM, y subí a 32GB</q>.

El tema es que al cambiar las RAM nos dimos cuenta que mi compu tenía un slot de RAM quemado y no podía aumentarse. Tuve que viajar a Santiago en `cuarentena` para cobrar la garantía, fue redificil. Lamentablemente, no habían más laptops de las características del mío y no había una solución rápida más que devolverlo. Me permitieron pedir la devolución del dinero y en modo urgente tuve que ir al *Parque Arauco* a llevarme un computador `ya`. 

No estaba el Legion 5, y un <mark>super vendedor</mark>, (se notaba que sabía) me metió un Legion 7 y me *"engrupió"* con que traía una RTX 2070. Y no me pude resistir, obviamente este Laptop salió harto más caro que el Legión 5. Entonces ahora sí necesitaba sacar un podio para poder pagarlo. El tema es que el compu no era con retiro inmediato sino que se envíaba a domicilio, y bueno, saben que la reputación de los envíos no es la mejor por la pandemia, no siempre llegan cuando dicen, pero este chico me *"re-contra-juró"* que en 4 días llegaba. Así que me la jugué, aproveché esos días para terminar mi memoria en un compu que tenía y afortunadamente el Legion 7 llegó en 3 días. Reinstalé todo, y entre la falla de mi Legion 5 hasta la puesta en marcha del Legion 7 pasaron cerca de 10 días sin poder modelar nada. Mantuve el primer lugar, lo cual me tenía preocupado, pero apenas me puse a probar nuevos modelos aparecieron los pesos pesados, casi todos los finalistas empezaron a superar el 50% y me tiraron muy atrás en el leaderboard.

## Otro Problema: 2da Lección

![picture of me]({{ site.urlimg }}concurso/codigo.jpeg){: .left .show-for-large-up .hide-for-print width="500"}
![picture of me]({{ site.urlimg }}concurso/codigo.jpeg){: .center .hide-for-large-up width="500"}
Un segundo problema, ya casi en la recta final del concurso fue, bueno, los modelos que generé fueron casi todos a la rápida, ¿cómo ordenarse? Y aquí viene la otra lección: <mark>MODULARIZACIÓN</mark>.

Comencé a Modularizar mi código y definir distintas partes lo más automatizadas posibles. Igual tenía una parte bien al lote, pero que me dio mucho valor, que fue el análisis exploratorio. Todo el resto del código lo dejé lo más ordenado que pude.

* **Generación de Variables**: Me preocupé de encapsular todos los procesos de creación de variables nuevas que requirieran gran proceso de cálculo en funciones sumamente flexibles. Si necesitaba generar más, sólo variaba parámetros y eso me permitía gran poder de experimentación, generando muchos cambios con sólo modificar algunos parámetros.

* **Código de Entrenamiento**: También me preocupé de modularizar esto, tratar de encapsular el preprocesamiento fue complejo, porque habían muchos formatos distintos dependiendo del modelo. Lo bueno fue que los modelos que fueron entregando los mejores resultados se empezaron a acotar y eso me permitió simplificar el Script.

* **Inferencia**: Esta parte se encargaba de poder generar las predicciones, pero además, generar una estrategia de ordenamiento de las predicciones que también era parte del problema. Finalmente, transformaba la predicción final en el formato requerido por la plataforma para el envío.

{% include alert success='Acá no descubrí nada nuevo, luego siguiendo a uno de mis Youtubers favoritos hoy, Abishek Thakur, y leyendo su [libro](https://github.com/abhishekkrthakur/approachingalmost/blob/master/AAAMLP.pdf), me dí cuenta que él propone un sistema muy similar, pero muchísimo más robusto que el mío.'%}

## Otra Lección: Aprender del error

Luego de muchos experimentos, noté que no estaba llevando registro de mis experimientos de una manera óptima. Lo peor de todo es que tengo un tutorial de MLflow, y <mark>NO LO UTILICÉ</mark>. Obviamente esto me llevó a notar que hay muchas herramientas para trackear experimentos, y hay una particular que me está cautivando harto: [Weigths & Biases](https://wandb.ai/).

Definitivamente noté un gran valor en llevar los resultados del mejor modelo, y eso lo hice. Pero también noté que hay un gran valor en llevar el registro de los experimentos no tan exitosos, porque me pasó que como probé muchas cosas, se me olvidaba qué cosas ya había probado. Eso me generó mucha inseguridad porque llegó un momento en la competencia que los puntajes se estancaron, en alrededor de un 55%, yo creo que iba 3ro y no tenía muchas más ideas. Y no sabía cuáles retomar y cuáles no.

{% include alert alert='Una dificultad que tenía esta competencia es que sólo indicaba el puntaje si éste era mejor que el anterior. En caso de no serlo, uno nunca sabía si había errado por poquito o estaba muy perdido.'%}

## Búsqueda de Hiperparámetros

![picture of me](https://optuna.org/assets/img/blog-multivariate-tpe.gif){: .right .show-for-large-up .hide-for-print width="500"}
![picture of me](https://optuna.org/assets/img/blog-multivariate-tpe.gif){: .center .hide-for-large-up width="500"}
Una de las cosas que para mí tiene mucho valor es el tema de entender qué hacen los hiperparámetros en un modelo. En general, leo mucho la documentación para saber qué significan, y tiendo a depender mucho de `GridSearch`. Voy mezclando esto con tuneo manual, para sacarle el jugo al modelo.

Esto tiene sus ventajas, tengo control total sobre la búsqueda de hiperparámetros, pero, depende de que yo esté atento a los resultados. Además GridSearchCV en Scikit-Learn es muy ineficiente, tiende a ser lento y crashear debido al alto consumo de RAM.

Esto me ayudó mucho a poner en práctica algunas cosas que aprendí hace tiempo, y que nunca había tenido que aplicar: `Downcasting` y el uso de `Matrices Sparse`, que en mi caso aplicaba. El uso de estas técnicas me ayudó mucho a reducir el tamaño del dataset de train y permitir que el GridSearch funcionara.

Para el futuro, pretendo utilizar un framework de Cross Validation más estático, al final es el KFold lo que aumenta mucho el costo de evaluación del modelo. Por lo tanto, pretendo aplicar estrategias más robustas y organizadas como las presentadas por [Abishek](https://www.youtube.com/watch?v=ArygUBY0QXw) acá.

Además, nunca me había dado el tiempo de aprender librerías específicas de optimización de Hiperparámetros, especialmente porque intenté aprender `hyper-opt` una vez y la encontré muy engorrosa e innecesaria. Obviamente, modelos grandes requieren una estrategia más Bayesiana para buscar hiperparámetros de manera más inteligente. Encontré que `Optuna` es una tremenda herramienta y la estaré agregando a mi arsenal.

## Soft Skills

![picture of me]({{ site.urlimg }}concurso/present.png){: .right .show-for-large-up .hide-for-print width="500"}
![picture of me]({{ site.urlimg }}concurso/present.png){: .center .hide-for-large-up width="500"}
Llegando ya a la última semana aparecieron los contendores pesados, y yo estaba sin ideas. [Joaquín](https://www.linkedin.com/in/joaqunv/) y [Malen](https://www.linkedin.com/in/malen-antillanca/) cada vez que subían un modelo me pasaban y se instauraron rápidamente en el primer lugar, y aparecieron tambíen el equipo de Rodrigo y [Paul](https://www.linkedin.com/in/paul-bertens-ai/) que hicieron pocas submisiones pero subían muy rápido. Entonces decidí ocupar todo lo que se me ocurría el día Jueves antes de finalizar la competencia, quedé en 2do lugar detrás de Joaquín y Malen envíando mi ultima submisión. Después de esto dije: <q>estoy cansado, esto es todo, con tal de presentar estoy contento</q>. Al levantarme el viernes ya iba 5to, y al final del día quedé 8vo, que fue mi puesto final, y estuve a punto de quedar fuera de la posibilidad de presentar.

{% include alert tip='Acá hay que ser muy estratégico, porque no supe como generar mis últimas submisiones. De hecho Paul y Rodrigo envíaron su submisión final a las 23:51 pasando en último momento a Joaquín y Malen. Entonces realmente hay que ser estratégico y pensar bien cuando quemar todos tus cartuchos.' %}

Para rematarme, ese día salió un reportaje en [`LUN`](https://www.lun.com/Pages/NewsDetail.aspx?dt=2020-12-11&PaginaId=24&bodyid=0) con los ganadores del año pasado, los fundadores de Acacia, Catalina Espinoza, y Abelino Jiménez. Ella, Magíster de la Universidad de Chile, él, Doctor en Carnegie Mellon, yo... un egresado de Ingeniería Civil 😔, hablaron de su solución con Redes Neuronales y lo primero que pensé fue... <q>uuuhhhh, no tengo oportunidad en el Pitch... a menos, que haga mi modelo interesante</q>. Y obvio, no todas las competencias se ganan así, pero afortunadamente esta sí. La habilidad de comunicar es tan importante como la habilidad para modelar. Creo que la competencia dejó en evidencia que varias personas saben hacer la pega... para más remate, Catalina y Abelino estaban entre los 10 finalistas y habían Estudiantes de Doctorado en AI, Profesores de Magíster, gente de Argentina, Perú, etc. ¿Cómo ganarles? Bueno, confiando en tu trabajo, entendiendo las fortalezas y debilidades de tu modelo y haciéndolo relevante. Para ello varios consejos:

* **Entendiendo el problema de Negocio**: Intuyendo para qué se usan estos modelos y poniendo atención a respuestas de los organizadores. Revisar los comentarios de la competencia me ayudó a entender cómo se iba a consumir el modelo, y cómo mi modelo podía ser de valor para las personas del banco.

* **Poniendo atención al Slack**: Mucha de las preguntas que otros competidores hicieron me ayudaron a mejorar mi modelo y a preparar mi Pitch final, leí todo el Slack. Además, leyendo sutiles pistas de los organizadores ayudaron a confiar en que mi modelo tenía potencial. A pesar de quedar 8vo, podía enfocarme en cómo mi modelo resolvía problemas reales que ellos tenían, como era el priorizar a qué clientes atacar.

![picture of me]({{ site.urlimg }}concurso/modelo.png){: .left .show-for-large-up .hide-for-print width="500"}
![picture of me]({{ site.urlimg }}concurso/modelo.png){: .center .hide-for-large-up width="500"}

* **Preparar el Pitch**: Son 4 minutos en los que te juegas la vida, hicimos 3... y es la peor experiencia por la que ha pasado. El corazón a mil y la verdad es que da mucho nervio. Si te trabas, no alcanzas a presentar todo en el tiempo dado y en verdad fueron super rigurosos con eso. Traté de incluir comentarios que se hicieron en Slack o durante la conversación de finalistas tratando de demostrar siempre cómo mi modelo suplía sus necesidades. 

La preparación de esos 4 minutos de Pitch me tomaba fácil 6 horas, y mucha práctica, realmente había que escoger muy bien qué decir y por sobre todo qué no decir. Para el último Pitch estaba muy cansado y la verdad es que el día anterior me fui a acostar sin tenerlo tan bien preparado. 

Pero bueno, el día de la final desperté a las 7am, y estuve 3 horas practicando el Pitch final (que era a las 10am). Después de mucho practicar no quedaba conforme, me trababa, se me olvidaban partes, no sé, entré a la presentación sin estar 100% convencido. 

Afortunadamente me fue bien, y cuando me avisaron que mi modelo era el ganador casi me da un ataque, pensé que me estaban bromeando. Fue bueno ver que el esfuerzo de años, aprendiendo por mi cuenta, no siendo considerado, o sencillamente siendo mirado en menos por la falta de mi título al final daba frutos.

Hay harta gente que agradecer en el camino: Mi esposa `Valentina`, que me dio permiso para acostarme tarde por más de un mes, y ella es muy friolenta. Pero desde la `Tam` que me avisó de la competencia, o amigos como `Lautaro`, yo creo que ni sabe, pero él me pasó el primer Libro de Machine Learning que leí, el [Applied Predictive Modeling](http://appliedpredictivemodeling.com/) de Max Kuhn, o amigos como Nico, que me hablaron de Deep Learning por primera vez, o Matías (el Dragon Slayer) que siempre me decía que tenía que usar Python y me presentó técnologías como Github o Pytorch. **Gracias!**

![picture of me]({{ site.urlimg }}concurso/LogoBinarioBlanco.png){: .right .show-for-large-up .hide-for-print width="500"}
![picture of me]({{ site.urlimg }}concurso/LogoBinarioBlanco.png){: .center .hide-for-large-up width="500"}

## Algunos comentarios finales:

* Es verdad que competir toma mucho tiempo, pero menos de lo que esperaba. Organizándose bien uno puede compatibilizar trabajo, familia y hobbies como este. Incluso en Kaggle.
* El computador es importante. Tener una buena maquina ayuda, pero si uno crea un código inteligente y flexible puede dejar entrenando en la noche y revisar resultados en el día. Esto me impactó mucho y estoy trabajando en un paquete en Python para facilitar esta parte. Espero pronto tener noticias, de un BETA.
* Practicar lo que predicas... hay que usar algún sistema de logging, es la única manera de tener tus experimentos ordenados. Por otra parte, si bien estrategias como RandomSearch o Gridsearch entregan buenos resultados, es necesario moverse a estrategias de búsqueda Bayesiana de modo de incrementar la eficacia de encontrar un buen set de hiperparámetros sin depender de la atención del modelador.
* Hay que entender bien qué modelos usar, y cuándo desertar de una idea que no da frutos. Esto lo veo porque los 3 finalistas utilizamos modelos de Gradient Boosting. Si bien, es posible utilizar otros, yo intenté lightFM y una red neuronal en `Keras` ninguno entró en el leaderboard. Esta parte es dificil, pero tuve que decidir que aunque los modelos se vieran prometedores, había que renunciar a ellos y seguir con las ideas que sí estaban dando resultado.
* Nunca menospreciar el poder de un buen Speech. Yo no soy bueno para hablar, ni me considero una persona buena para vender ideas. Pero sí soy seguro de lo que sé hacer. Y eso hay que trasmitirlo. Creo que la razón por lo que pude ganar es porque encontré la necesidad del banco, lo que ellos realmente esperaban del modelo y me aferré a eso y lo defendí las 3 rondas de Pitch y resultó.

# RECOMPENSAS

El premio es bueno (muuuuuuuy bueno), no se puede desmerecer. Pero la competencia significó mucho para mí en términos profesionales. No sólo se puede decir que, gracias al modelo, se pudo hacer un aporte super concreto para el beneficio de una empresa de bastante prestigio como lo es [Banco Itaú](https://banco.itau.cl/). Pero también, esto llegó a gente que de una u otra manera apuesta por ti. El hacer las cosas bien, o el mostrar que sabes hacerlas (y esto lo digo con infinita humildad, porque sé que aún hay mucho que tengo que recorrer) tiene sus frutos. Desde **Abril 2021** estoy trabajando con la gente de [Jooycar](https://www.jooycar.com/es/inicio/) como  <mark>Head de Data Science</mark>. No sé si estoy listo, no sé si soy el idóneo, pero vamos a dar lo mejor de uno para poder generar valor a partir de los datos. Tengo hartas expectativas de lo que podemos haer y agradezco mucho a la gente de [Innspiral](https://www.linkedin.com/company/innspiral/?originalSubdomain=cl) por dar la plataforma para mostrar lo que uno hace y por crear el Desafío Binnario.

### Acerca de Jooycar

Para los que no saben, Jooycar es una Startup Insurtech que se dedica a asegurar vehículos utilizando IoT para mejorar el comportamiento de conducción. Obviamente tremendos desafíos se vienen acá. Hay harto que hacer, y mucho que armar, pero de las cosas entretenidas que se vienen:

* Trabajo con **the real** Big Data.
* Modelos entretenidos, al menos planeo uno que otro modelito de Deep Learning (No tengo idea si sé hacerlos, muy probablemente no, pero tengo muchas ideas de lo que ya se puede venir).
* Trabajo con data geográfica (Algo nuevo para mí).
* Integración de Data Science con Diseño de Software (algo que hace rato estaba buscando).
* Y MUCHA INNOVACIÓN...!!!

Y bueno, espero que esto pueda ser de utilidad para quienes se inician y los que ya llevamos más tiempo en la Ciencia de Datos, y ojalá cuando haya una nueva competencia de [Binnario](https://binnario.ai/) (que de seguro vamos a estar ahí), todos podamos ser mejores y que el ganador siga con esta idea de compartir su experiencia.

Espero poder estar constantemente compartiendo material que sea útil para quienes son entusiastas del Machine/Deep Learning, y si en algo les puedo apoyar no duden en contactarme.

[**Alfonso**]({{ site.baseurl }}/contact/)





