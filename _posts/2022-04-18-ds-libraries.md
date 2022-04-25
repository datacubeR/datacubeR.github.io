---
permalink: /ds-tech/ 
title: "Qué debo aprender para ser Data Scientist."
subheadline: "Un compendio más de 100 tecnologías para Data Scientist."
teaser: "Un compendio de todas las tecnologías que he usado como Data Scientist."
# layout: page-fullwidth
usemathjax: true
category: ml
header: no
image:
    thumb: libraries/librerías.png
tags:
- python
- tutorial
published: false
---

La ciencia de datos es una de las disciplinas más de moda hoy en día. Y cómo que por alguna razón todos quieren ser parte de ello. Sin duda en el mediano/largo plazo probablemente todas las disciplinas tendrán una componente de datos y la verdad es que vale la pena aprender a lidiar con ellos.<!--more-->

Hoy en día la decisión es simple, trabajar con R o con Python, pero el tema es que Python tiene 150.000+ librerías y R tiene otras tantas, por lo que a veces es abrumante pensar, tengo que aprender todo? Si es que no, por donde empiezo, tengo un montón de opciones y no me gustaría perder el tiempo en cosas que no valen la pena.

Además, en plataformas como Linkedin siempre hay gente que se pone Data Science / Machine Learning / Analytics Expert y un largo etc. que probablemente en su vida a programado y comparten publicaciones como esta:

## TOP 10 LIBRERÍAS DE PYTHON
![picture of me]({{ site.urlimg }}libraries/ble.png){: .center}

Colocan una foto llena de logos, y un listado con nombres casi aleatorios:
    > Esta es una lista que encontré por ahí:

1. Pandas.
2. NLTK.
3. Plotly.
4. Scikit-learn.
5. Category-encoders.
6. Imbalance Learning. (Esta no es ni siquiera una librería en Python, se llama Imbalance-Learn)
7. XGBoost.
8. Keras / Tensorflow.
9. Theano. (Nadie usa esto ya)
10. Beautiful Soup.

Y uno dice, ya pos tengo que empezar a aprender. Y la verdad es que si bien son librerías que pueden ser útiles, hay que ver si realmente son aplicables al trabajo que haces y si vale el esfuerzo de aprenderlo.

Bueno trabajando como Data Scientist creo que he usado 100+ librerías, por lo que quiero hablar de cada una de ellas y dar mi opinión si vale la pena aprenderla o no. Quiero decir que en verdad llevo más tiempo usando R (cerca de 5 años) que Python (3 años), por lo que voy a tratar de dar mi opinión de ambos. 

### ¿Cómo nace la idea?

La idea nace porque siempre me pongo a rabiar cuando gente X publica algo copiado de plataformas como Coding Dojo, Datacamp, etc. con información incompleta y recomendando librerías que nunca han usado (bueno yo también voy a hacer eso). Entonces decidí que quiero hacer un compendio de las librerías más famosas que uno conoce en Data Science. 

El compendio incluirá lo siguiente: 

- Todas las librerías/tecnologías que he utilizado previamente. 
- Sólo en ocasiones excepcionales listaré librerías que no he utilizado. Sólo mencionaré librerías que no he usado en los siguientes casos:
    - Están en mi lista de estar próximo a usarla y si bien no tengo proyectos con ellas ya me he adentrado en su documentación.
    - Son demasiado famosas para dejarlas fuera.

Principalmente mencionaré librerías de Python, porque es el estado del arte en Ciencia de Datos y algunas librerías de mi tiempo usando R. 

Además me dí la lata de recorrer los 5000 paquetes más descargados en PyPI para recomendar librerías de Python, por lo que en el caso de que corresponda indicaré el Ranking y el número de descargas al 01-04-2022. Debo advertir que puedo estar un poco desactualizado en R porque dejé de usarlo definitivamente desde fines del 2020. Además cuando corresponda voy a mencionar otras tecnologías fuera de R y Python que quizás vale la pena conocer cuando se trabaja en ciertas áreas de la Ciencia de Datos.

- Librerías de Python incluirán Ranking (Rk) y número de descargas (ND).
- Librerías de R irán acompañadas de un indicador (R).
- Otras Técnologías que no son librerías ni de R ni de Python llevarán una (O) de Otras.


Voy además dividirlas en Prioridades I, II y III:
- I: Definitivamente debes aprenderlas ya. En el caso de R debes aprenderla ya, pero sólo si usas R.
- II: Dependiendo del caso (si trabajas con tecnologías anexas) podría ser una buena opción.
- III: No pierdas tu tiempo en aprenderlas. A lo más una que otra función puede ser útil en algún momento de la vida.

Finalmente, dividiré todas las recomendaciones en la siguientes categorías:
- Manipulación de Datos, 
- Bases de Datos, 
- Machine Learning, 
- Deep Learning, 
- Otras. 

{% include alert todo='Esta lista no es exhaustiva y si alguien quiere contribuir ayudando a reclasificar esto estoy abierto a sugerencias y colaboraciones.' %}

> Disclaimer: Todas las librerías que mencionaré son excelente en lo que hacen. Si recomiendo no aprenderlas no es porque sean malas (a menos que lo diga), es sólo que muy rara vez necesitarás utilizarlas debido a que son demasiado específicas.


Finalmente el objetivo final de este compendio es que los nuevos Data Scientist (y también los más experimentados) puedan tener una opinión de qué librerías existen y cuáles sí o sí deberían saber.

# Manipulación de Datos

- SQL (O, Pr: I): Si bien esta no es una librería de Python/R, esto es por lejos lo primero que todo Data Scientist debe saber. No es necesario ser un ultra experto en este tema pero sí al menos debes manejar los siguientes aspectos:

- SELECT/FROM
- JOINS: Entender las principales diferencias entre LEFT, RIGHT, INNER, SELF JOINS.
- WHERE, GROUP BY, HAVING.
- ORDER BY
- MIN,MAX, AVG, etc.
- CREATE (volatile, temporary) TABLES, INSERT INTO, WITH (Esto es bien difuso ya que depende del motor).
- Entender al menos los motores más populares que son por lejos MySQL y Postgresql.

- Pandas (Rk: 28, ND: 78M+, Pr: I): Esta es por lejos la librería más utilizada en Ciencia de Datos y para mi gusto la más completa. No está en el primer lugar porque realmente creo que es más importante saber SQL primero ya que es mucho más simple. Básicamente Pandas es un SQL con Esteroides, muchísimo más poderosa y que bajo ningún motivo puede ser reemplazada por SQL. Pero tiene tantos comandos que al principio uno podría no saber cómo empezar. Su API es tan buena que existen muchos mirrors, como Dask, koalas, o cuDF, que siguen la misma convención sólo que el backend hace algo distinto (Básicamente aprendiendo pandas se pueden aprender varias librerías a la vez). Mi recomendación es aprender cómo reproducir todo lo aprendido en SQL y luego aprender funciones para resolver problemas específicos. ¿Cómo aprender? Lo mejor es a través del [User Guide](https://pandas.pydata.org/docs/user_guide/index.html) en su propia documentación.

- Numpy (Rk: 28, ND: 78M+, Pr: III): Numpy es una librería de computación científica, esto quiere decir, computar/calcular implementaciones matemáticas/estadísticas desde test de hipótesis, Transformadas de Fourier, y un largo etc. Normalmente se recomienda aprender antes o junto a Pandas, pero realmente creo que (prepárense) <mark>no vale la pena aprenderla inicialmente</mark>. Hace unos años era necesario aprender numpy para complementar pandas, ya que habían muchas cosas que no estaban disponibles en pandas pero sí en Numpy, pero si es que no vas a hacer implementaciones directamente de Algebra Lineal, no va a ser necesario usarla. Obviamente cuando uno es avanzado se dará cuenta que es bueno entender conceptos de Numpy como la vectorización. Mi recomendación es aprender sólo funciones que no están en pandas a medida que las vayas necesitando.

- Scipy: Este es un pedacito de Numpy aún más específico. Definitivamente <mark>no vale la pena aprenderlo</mark>, y sólo se necesitarán funciones muy específicas. En mi caso sólo la he usado para utilizar matrices sparse cuando queremos disminuir el tamaño de matrices con demasiados ceros y cuando enseñé probabilidad, porque tiene todas las distribuciones de probabilidad (incluso si son muy raras) y tienen funciones para muestreos, pmf, pdf y cdf.

- dplyr ( (R), Pr: I): Diría que es la versión en R de pandas, pero es un poco más limitado. No porque no tenga las capacidades para hacer lo que pandas hace sino porque el ecosistema de R está disperso en más paquetes. Para emular pandas en R se tiene que usar casi todo el tidyverse: `dplyr`, `tidyr`, `lubridate` y `hms` (para fechas), `forecats` (para variables categóricas), `purrr` (para loops eficientes), `readr` + `vroom` para io, `stringr` y `stringi` para lidiar con strings. Creo que el uso del pipe (%>%) hace que el código en R sea más expresivo que en pandas y realmente vale la pena aprender este ecosistema si trabajas en R ya que es mucho más amigable que la sintaxis de R puro.

- Dask(Rk: 390, ND: 5.6M+, Pr: III): Corresponde al motor que provee paralelismo para Pandas. La librería es excelente pero bajo ningún motivo vale la pena invertir tiempo acá, porque básicamente es la misma interfaz de pandas. Basta con hacer `import dask.dataframe as dd` y anteponer dd en vez de pd y listo. No he tenido que usar nunca esta librería pero es demasiado famosa para no mencionarla.

- data.table (R principalmente pero creo que también está en Python, Pr: III): Este es un tema polémico porque hace mucho tiempo había una discusión entre el creador de esta librería y la gente de RStudio. Básicamente `data.table` es la librería más rápida para manejo de datos en R pero su sintaxis no es muy amigable. Afortunadamente Hadley Wickham creo dtplyr que permite usar data.table como el backend de dplyr, por lo que diría que si bien esta librería es extremadamente poderosa no vale la pena aprenderla.

- cudf (Rk: NA, ND: NA, Pr: III): cuDF es una librería que es parte de RAPIDS, un set de paquetes en Python desarrollado por NVIDIA que permiten ejecutar todo en GPU. Este es el mirror de Pandas, básicamente la misma sintaxis que pandas pero que en el backend se ejecuta en GPU. <mark>No vale la pena apenderla, ya que es igual a pandas</mark>.

- cupy (Rk: NA, ND: NA, Pr: III): Es el Mirror en este caso de Numpy. Si sabes Numpy entonces sabes cupy, no debería estar dentro de tus prioridades como Data Scientist. Pero en el caso de querer lanzar tus procesos a la GPU es excelente.

{% include alert alert='Estas librerías no deberían ser la mejor opción para trabajar con grandes volumentes de datos. Esto porque normalmente la GPU tiene menos RAM, a menos que tengas varias GPU o una RTX3090. La mayoría del tiempo utilizar pandas va a ser más que suficiente.'%}

- pyspark (Rk: 148, ND: 18.7M+, Pr: III): Este es la librería por excelencia para trabajar con Big Data. `pyspark` es el cliente de Python para el Spark de Scala. Lo bueno de esta librería es que te da la opción de usar una API muy similar al Spark en Scala o incluso una que utiliza comandos tipo SQL. Esta va a ser la mejor opción para cuando tengas que trabajar con Big Data y computación distribuida en un Cluster, pero <mark>NO VALE LA PENA APRENDERLO</mark>. Principalmente porque la interfaz de SQL te servirá la mayor cantidad del tiempo para llevar a cabo ETLs y en caso de procesamiento más rebuscado `koalas` es un mirror de pandas para ejecutar Spark. 

- koalas (Rk: 866, ND: 1.8M+, Pr: II): Si tienes que usar Spark yo creo que es mejor `koalas`, que tiene la sintaxis de pandas que uno ya sabe.

- sparklyr ((O), Pr: II): La única vez que tuve que trabajar con data en Spark fue en Python y usé koalas. Pero vale la pena mencionar esta librería porque básicamente permite ejecutar Spark usando sintaxis de `dplyr`. Si es que llegaras a necesitar Spark, mi opción recomendación sería hazlo en otro lenguaje (principalmente por los problemas de memory leakage de R) pero si necesitas hacerlo en R, esta es la mejor opción.

- Microsoft Excel ((O), Pr: I): Excel **nunca** debería ser una opción para trabajar con Datos, pero sí o sí tienes que saber usarlo porque lamentablemente los archivos `.xlsx` son todavía un formato extremadamente popular. <mark>NUNCA</mark> deberías utilizar Excel si no es sólo para entregar resultados. Si tú eres de los que aún dice que hay cosas que son más sencillas en Excel que en Pandas o SQL, es que no sabes utilizar bien esas tecnologías aún.

# Bases de Datos

- sqlalchemy (Rk: 46, ND: 44M+, Pr: I): Esta es por lejos una de las mejores librerías que se han creado en Python. Básicamente permite utilizar cualquier Base de Dato SQL con una interfaz común. Debo decir que si bien esta es una librería extremadamente poderosa y que vale completamente la pena aprender, la documentación está pensada para gente bien "computín" y no es tan amigable. Mi recomendación para aprenderla es mediante videos tutoriales. Ahora en Ciencia de Datos la vas a ocupar sí o sí si eres Data Engineer para poder modelar Base de Datos o hacer consultas. Como Data Scientist normalmente sólo la usarás como forma de conexión con Pandas mediante `create_engine` y `.to_sql()` para extraer datos.

- sqlmodel (Rk: NA, ND: NA, Pr: II): Esta es una librería creada hace muy poco por el gran [Sebastián Ramírez](https://www.linkedin.com/in/tiangolo/?originalSubdomain=de) (Tiangolo). No he utilizado esta librería pero sí sé que está construida sobre sqlalchemy. sqlmodel es a sqlalchemy lo que FastAPI es a Flask, por lo tanto, es muy posible que en el tiempo esta librería venga a reemplazar a SQLAlchemy principalmente porque Tiangolo dedica mucho tiempo a la buena documentación y casos de usos, cosa que SQLAlchemy no tiene tan bien hecho en mi opinión.

- DBI ((R), Pr: I): DBI viene a ser una interfaz comun para poder consultar datos. Creo que podría considerarse el simil de sqlalchemy, pero no sé si tiene tantas funcionalidades. Al menos esta siempre fue mi opción para conectarme a DBs en R, pero nunca me tocó modelar una base de datos como sí tuve que hacerlo en Python. DBI tiene conexión con casi todos los motores de SQL o usando conexión odbc.

- PyMongo (Rk: 142, ND: 19.5M+, Pr: II): Esta es la interfaz para utilizar MongoDB desde Python. MongoDB es probblemente la base de datos no relacional más famosa. Sólo vale la pena si es que te toca trabajar con MongoDB pero lo bueno es que su uso es sumamente intuitivo. Utiliza la misma sintaxis que MongoDB pero en vez de usar el formato BSON (que es como un tipo de JSON), lo hace en los diccionarios de Python. Y por cierto, hacer queries en MongoDB es básicamente SQL con otra sintaxis y permitiendo data no estructurada como output, por lo que aprenderla es bastante sencillo.

- elasticsearch-dsl (Rk: 808, ND: 1.9M+, Pr: II): Este no es la librería más popular para conectarse a ElasticSearch, que es un motor de base de datos basado en documentos que es extremadamente rápida. La sintaxis en ElasticSearch es horrible, y yo reconozco que no tengo idea como extraer datos usando ElasticSearch puro. El tema es que elasticsearch-dsl es tan intuitivo que pude generar procesos de ETL en ElasticSearch utilizando esta librería, ya que su API es como estilo dplyr (aunque es una librería de Python), lo que le permite ser muy expresiva y fácil de crear, leer y entender. Si alguna vez tienes que trabajar con ElasticSearch, usa esta librería ya que es muchísimo más sencilla.

- psycopg2 (Rk: 103 y 161, ND: 25.4M + 17M, Pr: III): El Ranking de esta librería es un poco extraño, la razón es porque si utilizas Windows descargas psycopg2, pero si tienes Mac o Linux descargas psycopg2-binary, por lo que en estricto rigor esta librería es la suma de ambos. Este es el cliente de Postgresql en Python, un motor de base de datos extremadamente popular y poderoso. Es una interfaz muy parecida a DBI en R. Es un cliente lowlevel y bien rápido para poder interactuar con DBs Postgres. Yo la he utilizado como motor tanto para DBs Postgres Puras o para Datawarehouse como Redshift que están basadas en Postgres. Además se puede conectar con `sqlalchemy`, por lo que diría que no es necesario aprender mucho su sintaxis porque saber `sqlalchemy` ya hace la pega.

- pyodbc (Rk: 158, ND: 17.3M, Pr: III): Es una librería que nos permite hacer conexiones ODBC. Esta librería la usé únicamente en Windows para conectarme con Teradata que es un motor de Base de Datos que suele ser utilizado en entornos de alta seguridad como Bancos o Retail (mi recomendación: no usen Teradata, funciona bien, es rápido y todo pero su documentación al no ser código abierto es pésima, por lo que cosas fáciles se pueden hacer pero encontrar cómo hacer algo fuera de la común es casi imposible. Se los dice alguien que lo usó por 5 años). Normalmente se utiliza una línea para conectarse y es compatible con `sqlalchemy`, por lo que no es necesario aprender mucho. (

- DBeaver ((O), Pr: II): Esta es un cliente de bases Open Source gratis (aunque también tiene una versión pagada). Básicamente es un software que puedes descargar que te permite conectarte a cualquier Base de Dato SQL e incluso en la versión paga te permite conectarte a MongoDB y ElasticSearch, entre otras. Es rápido, tiene posibilidad de tener los modelos ER de cada Esquema además de varios atajos de teclado. Muy buena opción para conectarse con distintos motores.

# Visualizaciones

Esta es probablemente mi parte más débil principalmente porque es un área que no me gusta. Aún así he usado varias librerías, las cuales voy a mencionar ahora.

- Seaborn (Rk: 306, ND: 7.6M+, Pr: I): Probablemente no esperaban que esta fuera mi primera opción. La razón por la que la menciono en primer lugar es porque es una librería con funcionalidades restringidas pero que hacen la pega muy bien. Tiene la mayoría de gráficos prehechos y permite sin mucho código hacer gráficos muy bonitos y muy expresivos. Mi recomendación es sólo aprender `sns.catplot()` que permite graficar gráficos de variables categóricas o combinación categórica numérica (conteos, barplots, boxplot y amigos, etc.), `sns.relplot()` que permite generar gráficos para variables sólo númericas (scatter, line plots) y `sns.displot()` que grafica básicamente histogramas. Estas 3 funciones tienen interfaz comunes con built-in facet y varias manera de agrupación (columnas, filas, colores, estilos, etc.). Una de las cosas que más me entusiasma es que Seaborn comenzó a desarrollar una interfaz muy similar a `ggplot2` de R lo cual la haría extremadamente flexible y fácil de usar. Definitivamente vale la pena aprenderla.

- Matplotlib (Rk: 113, ND: 23.3M+, Pr: I): Yo creo que el ranking es un poco mentiroso, principalmente porque matplolib es dependencia de casi todas las librerías de gráficos, por lo que siempre la vas a necesitar. Lamentablemente hay que aprenderla. Y digo lamentable, porque a pesar de ser muy poderosa, considero que la documentación es como engorrosa y tiene una sintaxis muy verbosa. Además `seaborn` está construida sobre `matplotlib`, por lo que en casos de querer cambiar elementos del layout en `seaborn` se debe hacer mediante comandos `matplotlib`. Mi recomendación es aprenderla con ejemplos y algún cursito corto en Datacamp, porque es realmente difícil de aprender (no por su sintaxis sino que porque tiene muchas maneras distintas de hacer lo mismo y que a veces aplican y otras veces no).

- ggplot2 ((R), Pr: I): Para muchos es la mejor librería de visualizaciones que existe. Y quizás tienen razón. `ggplot2` es un remake de ggplot (que fue un fracaso) y que está basado en el grammar of graphics que es un concepto en el cual las partes del gráfico se construye en capas (la figura; ejes; elementos como puntos, líneas, boxplots; cálculos como regresiones lineales, promedios, intervalos de confianza; etc.) Además como que por defecto la paleta de colores y los ejes son bien bonitos. Yo considero que no es tan fácil de aprenderla pero es la mejor sintaxis para graficar.
Existen algunas librerías/addins en RStudio como `esquisse` que permiten crear ggplots (te entrega el código incluso) con una interfaz tipo Tableau. Muy recomendada si trabajas en R y/o en Python. Además tiene un enorme ecosistema de librerías complementarias para poder graficar casi cualquier cosa.

- plotnine (Rk: 2419, ND: 232K+, Pr: III): Es la versión en Python de ggplot2. Creo que es un tremendo esfuerzo y casi todas las funcionalidades están implementadas pero no funciona tan bien como ggplot2 (su ranking lo indica). El problema es que ggplot2 tiene muchos paquetes que lo complementan. Uno de los más poderosos es `patchwork` que es una interfaz para crear gráficos sobre gráficos de manera muy sencilla. Este es precisamente uno de las grandes problemáticas de plotnine, si se quieren crear layouts un poco más complejos comenzamos nuevamente a depender de `matplotlib` lo que evita una sintaxis única. Gracias a ver visto un EDA por [Martin Henze](https://www.linkedin.com/in/martin-henze/) utilizando ggplot comencé a usar esta librería pensando que podría lograr los mismos resultados, pero lamentablemente ggplot es muy superior.

{% include alert info='En mi opinión el 90% del tiempo utilizar gráficos estáticos será más que suficiente tanto para compartirlos en un PPT o para hacer EDAs. En caso de crear alguna aplicación interactiva entonces gráficos dinámicos e interactivos como los que hacen las siguientes librerías son una buena opción.'%}


- plotly (Rk: 315, ND: 7.3M+, Pr: III): Plotly es una librería basada en D3, que a su vez es una librería de Javascript que se hizo muy popular gracias a su capacidad de desarrollar gráficos interactivos muy bonitos. Hoy tiene APIs en casi todos los lenguajes más populares. Para mí gusto es una librería que sólo vale la pena aprender si es que estás completamente dedicado a las visualizaciones. Si bien es una librería poderosa es muy verbosa. Afortundamente paquetes como `plotly-express` han aparecido para abstraer la verbosidad y crear versiones de gráficos comúnmente usados en pocas líneas.

- plotly-express (Rk: 2802, ND: 167K+, Pr: II): Es la versión menos verbosa de plotly, si bien es un pelín menos poderosa debido a que es más simple, la mayor parte del tiempo será maś que suficiente.

- altair (Rk: 381, ND: 5.6M+, Pr: III): Es otra librería muy parecida a Seaborn en términos de sintaxis pero con la interactividad de plotly. Yo la utilicé sólo una vez creando una app en Streamlit. La razón: no quería usar plotly (en ese tiempo no conocía plotly express) y quedaban los gráficos más bonitos que en matplotlib y seaborn que eran estáticos. No vale la pena aprenderla y rara vez la verán por ahí.

- bokeh (Rk: 381, ND: 5.6M+, Pr: II): Es otra librería proveniente de Javascript que puede ser usadas desde R o Python. La verdad es que no la he usado, pero pueden ser alternativas para plotly ya que también son interactivas basadas en HTML pero con una sintaxis más simple. Nuevamente las recomiendo sólo en caso de dedicarse el BI o al Data Storytelling donde vale la pena invertir en visualizaciones llamativas.

### Otras herramientas BI

- Tableau ((O), Pr: II): En el caso de trabajar en Business Intelligence donde el foco es más mostrar herramientas interactivas que puedan manipular la data con algunos clicks, aparecen herramientas que no están basadas en código. Tableau es una muy buena alternativa. Es rápido, fácil de crear Dashboard con gráficos que sirven como filtros y pueden interactuar entre ellos. El problema, es que su costo es prohibitivo, su licencia es extremadamente cara y hoy existen otras herramientas más baratas que hacen lo mismo.

- PowerBI ((O), Pr: II): Es el Tableau de Microsoft. Es una buena alternativa con costos de licencias bastante más bajo. Sigue la misma idea de Tableau de usar cajitas tipo Pivot Tables para crear gráficos. Igual de eficiente que Tableau pero mucho más barato.

- Qliksense ((O), Pr: II): No recuerdo quien creo esto, pero es otra versión. Funciona exactamente igual que los otros dos. Tienen las mismas funcionalidades. Ninguna ventaja ni desventaja con los otros. ¿Cuél elegir? Da lo mismo, es lo que tu empresa esté dispuesta a pagar.

- Shiny ((R), Pr: I): Podríamos decir que es la versión en R de estos productos. La diferencia es que es gratis, y es basado completamente en código. Permite crear todo tipo de Dashboards interactivos mezclando cualquier otra librería de R (aunque también se podría agregar Python mediante `reticulate`) tanto para manipular datos como para visualizar. Es extremadamente poderosa y flexible y hay varias empresas que crearn sus portales utilizando Shiny. El problema es que no es tan fácil de hostear. En mi tiempo sólo RStudio ofrecía servicios para hostear ShinyApps (algunos gratis y otros de pago). Lo bueno es que se comenzó a crear todo un ecosistema en torno a Shiny, el cual tiene temas (basados en Bootstrap, material y otros frameworks de HTML, CSS y Javascript), hay una librería llamada golem, que permite modularizar grandes aplicaciones e incluso se permiten ingresar elementos nativos en HTML, CSS o Javascript. Vale completamente la pena aprenderlo <mark>si es que</mark> te dedicas al BI en R y tienes tiempo de crear todo desde cero. Va a ser más flexible que Tableau, PowerBI o Qliksense, pero hay que crear todo.

- streamlit (Rk: 1210, ND: 958K+, Pr: I): Similar a Shiny pero en Python. En mi opinión es mucho más sencillo de utilizar, pero mucho más simplista. Tiene lo justo y necesario para hacer funcionar una excelente aplicación demo. Lo bueno es que Streamlit fue comprado por HuggingFace por lo que se ha estado llevando sus funcionalidades para que sea el front-end de modelos de Machine Learning. Una ventaja de streamlit es que es fácilmente hosteable en cualquier servidor con Python (que son casi todos), en Heroku, en un servicio provisto por la misma gente de Streamlit o en Huggingface Spaces, siento estos últimos totalmente gratis. En el caso de querer hacer una demo, se puede crear algo de gran calidad y complejidad en no más de una hora. Su sintaxis es muy sencilla y se puede aprende en unas horas.

- Dash (Rk: 1263, ND: 885K+, Pr: III): Este es casi idéntico a Shiny (pero también en Python). Yo lo usé sólo una vez en un proyecto, y no nos gustó porque era muy complicado de setear. Básicamente crear el CSS que dejara los distintos divs en orden fue un martirio por lo que siempre nos quedaba la aplicación descuadrada. No vale la pena, ya que streamlit simplificó esto infinitamente.

- Gradio (Rk: 3481, ND: 104K+, Pr: II): Es una interfaz aún más simple que Streamlit, pero con muchas menos funcionalidades. Esta librería sí que se creó con el sólo propósito de ser un IO para modelos de Machine Learning. A diferencia de Streamlit que puedes crear Dashboards, sitios webs, agregar gadgets, Gradio sólo le interesa crear gadgets de input/output para un modelo a modo de demo. Yo no lo he usado aún, pero decidí aprenderlo luego de ver una demo de un Pipeline de Transformers por Omar Sanseviero, donde construyó un front-end con modelos de Generación de Texto y Machine Translation en 10 mins. Puedes ver su presentación [acá](https://www.youtube.com/watch?v=Mg7YeWBUKbM). Vale mencionar que también fue adquirido por HuggingFace por lo que puedes hostearlo facilmente en servidores Python, Heroku o Spaces.

- Django (Rk: 319, ND: 7.06M+, Pr: III): No lo he usado. Pero es por lejos la librería más poderosa de desarrollo Web. Acá ya no hablamos sólo de una interfaz de Dashboards sino que un software completo. Es tanto así que existen Ingenieros de Software especializados sólamente en el Ecosistema Django. Por nada del mundo como Data Scientist debieras tener que llegar a usar una librería tan poderosa como esta. Pero si te interesa crear una aplicación a nivel profesional con procesos de datos o Modelos de Machine Learning por abajo, esta podría ser una opción. Algunas aplicaciones creadas en Django son Instagram, Spotify, Youtube, Dropbox, entre otras.

- Flask (Rk: 86, ND: 30.9M+, Pr: III): Tampoco lo he usado, pero tengo entendido que es un Django pequeñito, que además tiene otras funcionalidades como crear APIs. Es aún extremadamente popular en entornos de desarrollo web, pero en mi opinión está poco a poco cayendo en desuso, principalmente debido a que FastAPI está ganando mucho protagonismo en cuánto a APIs se refiere y es una opción mucho más sencilla de aprender.


# Machine Learning

Esta es por lejos mi sección favorita, por lo que puede que me extienda un poco más de que el resto.

- Scikit-Learn (Rk: 92, ND: 30.1M+, Pr: I): Es la librería por excelencia para crear modelos de Machine Learning. La sintaxis de su API está tan bien diseñada que una manera de reconocer que otras librerías de Machine Learning son confiables es si es que siguen su API. Básicamente `scikit-learn` es super reconocida por sus modelos como Clase y su estandar `fit-transform-predict`, además de casi 15 años de vida. Si quieres hacer modelos de Machine Learning sí o sí tienes que partir por acá por varias razones: (1) Su documentación es excelente, incluso puedes aprender la teoría detras de cada modelo leyendo su [User Guide](https://scikit-learn.org/stable/user_guide.html) (toda persona que se dedique al ML debería leer la documentación completa de Sklearn una vez al año 🤪). Además contiene sólo modelos ML que están en el estado del arte. De hecho para que un modelo se implemente en Scikit Learn tiene que cumplir [requisitos](https://scikit-learn.org/stable/faq.html) muy estrictos. Este es por lejos una de las mejores inversiones que uno hará como Data Scientist, ya que aprendiendo a utilizar esta librería podrás utilizar millones de otras basadas en la misma API.

- tidymodels (R, Pr: II): Yo solía ser un fan de esta compilación de librerías. Creo que Max Kuhn es un tremendo desarrollador y lo respeto profundamente, pero creo que parsnip trató de llevar el modelamiento en R a un estado incluso más flexible que `scikit-learn` pero no les funcionó. Lamentablemente el Machine Learning en R está disgregado en muchas librerías todas con APIs diferentes, por lo que este esfuerzo de unificar todo es increíble. Lamentablemente el memory leakage que sufre R y el tremendo trabajo de los mantenedores de `scikit-learn` hacen que un esfuerzo como este nunca logre la popularidad que tiene Python en este rubro. Tidymodels está basado en 3 paquetes principalmente: `recipes`, para el preprocesamiento, que a mi gusto tiene una mucho mejor API que Scikit, `parsnip`, que es la unificación de todos los modelos de ML implementados en R y `yardstick` que contiene todas las métricas de evaluación. Si te dedicas a hacer modelos pequeñitos de prueba, sin mucho intensidad de cómputo es una opción, en cualquier otro caso vale más cambiarse a `scikit-learn`.

- caret (R, Pr: II): Este es el predecesor de `tidymodels`. A pesar de ser una librería que se le quitó mantenimiento hace un tiempo sigue disfruntando de mucha popularidad ya que tiene más de 200 modelos implementados. El propósito de Caret es el mismo de tidymodels sólo que su API no era compatible con el tidyverse por lo que decidieron seguir el esfuerzo de tidymodels. Este proyecto contaba con todo integrado, preprocesamiento, entrenamiento, postprocesamiento, esquemas de validación, métricas de evaluación, incluso ensambles. Por alguna razón lamentablemente decidieron cortarlo.

- pycaret (Rk: 1833, ND: 418K+, Pr: III): Este es un proyecto en Python que nace de la base de Caret y que se ha hecho extremadamente popular. En mi opinión sólo vale la pena aprenderlo si es que no te gusta codear. Las ventajas es que permite hacer mucho en pocas líneas de código y es compatible con muchas librerías externas como XGBoost, LightGBM, etc. Además cuando uno no es expertos en tareas menos habituales como Anomaly Detection o Series de Tiempo permite seguir el mismo esquema de código. Lo que me gusta del creador de esta librería es que el deja muy en claro que su objetivo que es que los Citizen Data Scientist pueden tener modelos de alta calidad a la mano. 






# todo
Manipulación de Datos, Bases de Datos, Machine Learning, Deep Learning, otras.




[**Alfonso**]({{ site.baseurl }}/contact/)
