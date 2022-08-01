---
permalink: /ds-lib/ 
title: "¿Qué debo aprender para ser Data Scientist?"
subheadline: "Un compendio con más de 100 tecnologías para Ciencia de Datos."
teaser: ""
# layout: page-fullwidth
usemathjax: true
category: ds
header: no
image:
    thumb: libraries/librerías.png
tags:
- python
- tutorial
published: true
---

![picture of me]({{ site.urlimg }}libraries/librerías.png){: .left .show-for-large-up .hide-for-print width="300"}
![picture of me]({{ site.urlimg }}libraries/librerías.png){: .center .hide-for-large-up width="250"}

La ciencia de datos es una de las disciplinas más de moda hoy en día. Y cómo que por alguna razón todos quieren ser parte de ello. Sin duda en el mediano/largo plazo probablemente todas las disciplinas tendrán una componente de datos y la verdad es que vale la pena aprender a lidiar con ellos.<!--more-->

Hoy en día la decisión es simple, trabajar con R o con Python, pero el tema es que Python tiene 150.000+ librerías y R tiene otras tantas, por lo que a veces es abrumante pensar, tengo que aprender todo? Ojo, eso sin contar otro tipo de tecnologías de Visualización, ETL y un largo etc. Por donde empiezo, tengo un montón de opciones y no me gustaría perder el tiempo en cosas que no valen la pena.

Además, en plataformas como Linkedin siempre hay gente en cuyo título dice Data Science \| Machine Learning \| Analytics Expert y un largo etc. y que probablemente en su vida ha programado y comparten publicaciones como esta:

# TL;DR
{: .no_toc }
*[TL;DR]: Too Long; Didn't Read



## TOP 10 LIBRERÍAS DE PYTHON
{: .no_toc }

Esta es una lista que encontré por ahí:

1. Pandas.
2. NLTK.
3. Plotly.
4. Scikit-learn.
5. Category-encoders (era tremenda librería pero está sin mantenimiento actualmente)
6. Imbalance Learning. (Esta no es ni siquiera una librería en Python, se llama Imbalanced-Learn)
7. XGBoost.
8. Keras / Tensorflow.
9. Theano. (Nadie usa esto ya)
10. Beautiful Soup.

Colocan una foto llena de logos, y un listado con nombres casi aleatorios:  
![picture of me]({{ site.urlimg }}libraries/ble.png){: .center}

A veces te indican qué librerías sí o sí tienes que saber, y nunca las has escuchado:

![picture of me]({{ site.urlimg }}libraries/libraries_1.jpg){: .center}

A veces tienen hasta errores burdos: 

![picture of me]({{ site.urlimg }}libraries/libraries_2.jpg){: .center}

Y uno se pregunta ¿Con qué parto?. Y la verdad es que si bien son librerías que pueden ser útiles, hay que ver si realmente son aplicables al trabajo que haces y si vale el esfuerzo de aprenderlo.

## Un Alto antes de Continuar
{: .no_toc }

La ciencia de datos es una disciplina enorme. Y hay que darla paso a paso, o no vamos a lograr nada y vamos a vivir estresados de tantas cosas que no sabemos usar y que tenemos que aprender. No digo que mi caso sea el perfecto, para nada, pero yo partí así:

* **Business Analyst** (Una especie de Data Analyst, pero enfocado en dar valor al negocio 🤭 ja ja): En mis primeros 2 años, lo que más hacía era responder preguntas con datos. El resultado, una query en SQL, la mayor parte del tiempo con una tabla exportada en Excel. Aprendí mucho SQL porque las Bases de Datos que usábamos eran gigantes y muy complejas. Responder una pregunta de negocio podía tomar 6 o 7 subqueries, con muchos joins en cada una de ellas. Luego tuve la oportunidad de crear algoritmos sencillos para aplicar lógicas de negocios, a esto le llamábamos Calculation Engines (Motores de cálculo). Y es básicamente aplicar lógicas de negocio complejas en los datos para chequear qué clientes cumplían o no regulaciones bancarias. Luego muté nuevamente a algo más BI, y me tocaba hacer dashboards en Tableau todo el día, todos los días. La data que el dashboard necesita no se ordena sola, por lo que aparte de hacer gráficos que digan algo, había que hacer mucho SQL de fondo. No fue hasta como mi tercer año de Analista que comencé a hacer una Regresión o un SVM loco por ahí. Todo esto en R.

* **Data Scientist**: Luego de como 4 años logre un puesto de Data Scientist. Ya llevaba como 1 año haciendo modelos a escondidas, porque no era mi rol. Y acá me cambié a Python definitivamente. Tuve que aprender mucho pandas, Scikit-Learn (y los 3 grandes XGBoost, LightGBM y CatBoost) y modelar mucho. Pero con muchos errores teóricos de fondo, y ahí decidí que era importante entender el transfondo teórico. En ese tiempo leía mucho blog y veía mucho video (aún lo hago, pero ahí partí). Quizás desde el 2021 que ya me metí de lleno en el Deep Learning y acá estamos.

> Todo tiene que ser progresivo. El `Deep Learning` es sólo una extensión del `Machine Learning`, en vez de hacer feature selection/engineering, acá hay que hacer "Architecture Engineering", tratando de encontrar la arquitectura más apropiada a un problema. Por otra parte el `Machine Learning` es una extensión del `Análisis`. En vez de que tenga que analizar la data manualmente, el modelo aprende los insights por mí y a escala, pero hay que entregar data estructurada. Y el `Análisis` es sólo una extensión de la `Manipulación de Datos`. Sólo se puede entender la data una vez que la tengo ordenadita. Entonces, hay que partir de a poco, y no saltarse pasos.

{% include toc.md %}

# La idea

Trabajando como Data Scientist creo que he usado 100+ librerías y otras tecnologías, por lo que quiero hablar de cada una de ellas y dar mi opinión si vale la pena aprenderla o no. Quiero decir que en verdad llevo más tiempo usando R (cerca de 5 años) que Python (3 años), por lo que voy a tratar de dar mi opinión de ambos.

La idea nace porque siempre me pongo a rabiar cuando <q>gente experta</q> publica algo copiado de plataformas como Coding Dojo, Datacamp, etc. con información incompleta y recomendando librerías que nunca han usado (y hoy yo también voy a hacer eso 🤭😅). Entonces decidí que quiero hacer un compendio de las tecnologías más famosas que hay relacionadas a la ciencia de datos. 

El compendio incluirá lo siguiente: 

* Todas las librerías/tecnologías que he utilizado previamente. 
* Sólo en ocasiones excepcionales listaré librerías que no he utilizado cuando ocurra algunos de los siguientes casos:
    * Están en mi lista de estar próximo a usarla y si bien no tengo proyectos con ellas ya me he adentrado en su documentación.
    * Son demasiado famosas para dejarlas fuera.

Principalmente mencionaré librerías de Python, porque es el estado del arte en Ciencia de Datos y algunas librerías de mi tiempo usando R. 

Además me dí la lata de recorrer los 5000 paquetes más descargados en PyPI para recomendar librerías de Python, por lo que en el caso de que corresponda indicaré el Ranking y el número de descargas al 01-07-2022. Debo advertir que puedo estar un poco desactualizado en R porque dejé de usarlo definitivamente desde fines del 2019. Además cuando corresponda voy a mencionar otras tecnologías fuera de R o Python que quizás vale la pena conocer cuando se trabaja en ciertas áreas de la Ciencia de Datos.

* Librerías de Python incluirán Ranking en PyPI (Rk) y número de descargas (ND).
* Librerías de R irán acompañadas de un indicador (R).
* Otras Técnologías que no son librerías ni de R ni de Python llevarán una (O) de Otras.

Voy además dividirlas en Prioridades:
* 1: Definitivamente debes aprenderlas y empezar a utilizarlas ya. En el caso de R debes aprenderla ya, pero sólo si usas R.
* 2: Dependiendo del caso (si trabajas con tecnologías anexas) podría ser una buena opción.
* 0: No pierdas tu tiempo en aprenderlas. No porque sea mala, sino que la vas a necesitar de manera muy esporádica, por lo que hay que saber qué puede hacer, para qué sirve y puede que en algún momento de la vida una que otra función sea útil.

Finalmente, dividiré todas las recomendaciones en las siguientes categorías:
* Manipulación de Datos, 
* Bases de Datos, 
* Machine Learning, 
* Deep Learning, 
* Misceláneo. 
* Librerías Estándar

{% include alert todo='Esta lista no es exhaustiva y si alguien quiere contribuir ayudando a reclasificar esto estoy abierto a sugerencias y colaboraciones.' %}

> Disclaimer: Todas las librerías que mencionaré son excelente en lo que hacen. Si recomiendo no aprenderlas no es porque sean malas (a menos que lo diga), es sólo que muy rara vez necesitarás utilizarlas debido a que son demasiado específicas y no vale la pena enfocarse en ellas. Basta con leer la documentación un rato antes de utilizarla y saber que existe.

Finalmente el objetivo final de este compendio es que los nuevos Data Scientists (y también los más experimentados) puedan tener una opinión de qué librerías existen y cuáles sí o sí deberían dominar.

# Manipulación de Datos

- **SQL** ((O), Pr: 1): Si bien esta no es una librería de Python/R, esto es por lejos lo primero que todo Data Scientist debe saber. No es necesario ser un ultra experto en este tema pero sí al menos debes dominar los siguientes aspectos:

* SELECT/FROM
* JOINS: Entender las principales diferencias entre LEFT, RIGHT, INNER, SELF JOINS.
* WHERE, GROUP BY, HAVING.
* ORDER BY
* MIN,MAX, AVG, etc.
* CREATE (volatile, temporary) TABLE, INSERT INTO, WITH (Esto es bien difuso ya que depende del motor).
* Entender al menos los motores más populares que son por lejos MySQL y Postgresql.

Es muy triste ver gente que se hace llamar Data Scientist y no sabe hacer una query. Sin datos, no hay Científico de Datos, por lo que sí o sí dale a esto primero que cualquier otra cosa!!

- **Pandas** (Rk: 31, ND: 86M+, Pr: 1): Esta es por lejos la librería más utilizada en Ciencia de Datos y para mi gusto la más completa. No está en el primer lugar porque realmente creo que es más importante saber SQL primero ya que es mucho más simple. Básicamente Pandas es un SQL con Esteroides, muchísimo más poderosa y que bajo ningún motivo puede ser reemplazada por SQL. Pero tiene tantos comandos que al principio uno podría no saber cómo empezar. Su API es tan buena que existen muchos mirrors, como Dask, koalas, o cuDF, que siguen la misma convención sólo que el backend hace algo distinto (Básicamente aprendiendo pandas se pueden aprender varias librerías a la vez). Mi recomendación es aprender cómo reproducir todo lo aprendido en SQL y luego aprender funciones para resolver problemas específicos. ¿Cómo aprender? Lo mejor es a través del [User Guide](https://pandas.pydata.org/docs/user_guide/index.html) en su propia documentación.

- **Numpy** (Rk: 15, ND: 110M+, Pr: 0): Numpy es una librería de computación científica, esto quiere decir, computar/calcular implementaciones matemáticas/estadísticas desde test de hipótesis, Transformadas de Fourier, y un largo etc. Normalmente se recomienda aprender antes o junto a Pandas, pero realmente creo que (prepárense) <mark>no vale la pena aprenderla inicialmente</mark>. Hace unos años era necesario aprender numpy para complementar pandas, ya que habían muchas cosas que no estaban disponibles en pandas pero sí en Numpy, pero si es que no vas a hacer implementaciones directamente de Algebra Lineal, no va a ser necesario usarla. Obviamente cuando uno es avanzado se dará cuenta que es bueno entender conceptos de Numpy como la vectorización. Mi recomendación es aprender **sólo funciones que no están en pandas** a medida que las vayas necesitando.

A varios les puede llamar la atención que tiene más descargas que Pandas, pero la explicación es sencilla. Muchas librerías tiene como dependencia Numpy, Scikit-Learn, Matplotlib, pandas, y un largo etc, que hace obligatorio siempre tenerla instalada.

- **Scipy** (Rk: 65, ND: 42M+, Pr: 0): Este es un pedacito de Numpy aún más específico. Definitivamente <mark>no vale la pena aprenderlo</mark>, y sólo se necesitarán funciones muy específicas. En mi caso sólo la he usado para utilizar matrices sparse cuando queremos disminuir el tamaño de matrices con demasiados ceros y cuando enseñé probabilidad, porque tiene todas las distribuciones de probabilidad (incluso si son muy raras) con sus respectivas funciones para muestreos, pmf, pdf y cdf.

- **dplyr** ((R), Pr: 1): Diría que es la versión en R de pandas, pero es un poco más limitado. No porque no tenga las capacidades para hacer lo que pandas hace sino porque el ecosistema de R está disperso en más paquetes. Para emular pandas en R se tiene que usar casi todo el tidyverse: `dplyr`, `tidyr`, `lubridate` y `hms` (para fechas), `forecats` (para variables categóricas), `purrr` (para loops eficientes), `readr` + `vroom` para io, `stringr` y `stringi` para lidiar con strings. Creo que el uso del pipe (%>%) hace que el código en R sea más expresivo que en pandas y realmente vale la pena aprender este ecosistema si trabajas en R ya que es mucho más amigable que la sintaxis de R puro.

- **Dask** (Rk: 390, ND: 5.6M+, Pr: 0): Corresponde al motor que provee paralelismo para Pandas. La librería es excelente pero bajo ningún motivo vale la pena invertir tiempo acá, porque básicamente es la misma interfaz de pandas. Basta con hacer `import dask.dataframe as dd` y anteponer `dd` en vez de `pd` y listo. No he tenido que usar nunca esta librería pero es demasiado famosa para no mencionarla.

- **data.table** (R principalmente pero creo que también está en Python, Pr: 0): Este es un tema polémico porque hace mucho tiempo había una discusión entre el creador de esta librería y la gente de RStudio. Básicamente `data.table` es la librería más rápida para manejo de datos en R pero su sintaxis no es muy amigable. Afortunadamente Hadley Wickham creo `dtplyr` que permite usar data.table como el backend de dplyr, por lo que diría que si bien esta librería es extremadamente poderosa no vale la pena aprenderla si sabes `dplyr`.

- **cudf** (Rk: NA, ND: NA, Pr: 0): `cuDF` es una librería que es parte de RAPIDS, un set de paquetes en Python desarrollado por NVIDIA que permiten ejecutar todo en GPU. Este es el mirror de Pandas, básicamente la misma sintaxis que pandas pero que en el backend se ejecuta en GPU. <mark>No vale la pena apenderla, ya que es igual a pandas</mark>.

- **cupy** (Rk: NA, ND: NA, Pr: 0): Es el Mirror en este caso de Numpy. Si sabes `numpy` entonces sabes `cupy`, no debería estar dentro de tus prioridades como Data Scientist. Pero en el caso de querer lanzar tus procesos a la GPU es excelente.

{% include alert alert='Estas librerías no deberían ser la mejor opción para trabajar con grandes volumentes de datos. Esto porque normalmente la GPU tiene menos RAM, a menos que tengas varias GPU o una RTX3090. La mayoría del tiempo utilizar pandas va a ser más que suficiente.'%}

- **pyspark** (Rk: 127, ND: 23.9M+, Pr: 0): Este es la librería por excelencia para trabajar con Big Data. `pyspark` es el cliente de Python para el Spark de Scala. Lo bueno de esta librería es que te da la opción de usar una API muy similar al Spark en Scala o incluso una que utiliza comandos tipo SQL. Esta va a ser la mejor opción para cuando tengas que trabajar con Big Data y computación distribuida en un Cluster, pero <mark>NO VALE LA PENA APRENDERLO</mark>. Principalmente porque la interfaz de SQL te servirá la mayor cantidad del tiempo para llevar a cabo ETLs y en caso de procesamiento más rebuscado `koalas` es un mirror de pandas para ejecutar Spark. 

- **findspark** (Rk: 729, ND: 2.4M+, Pr: 0): Tan enredada es la instalación de Spark que se creó una librería para tener el path de instalación y poder levantar un Cluster local. Sólo sirve para eso.

- **koalas** (Rk: 1047, ND: 1.4M+, Pr: 0): Si tienes que usar Spark yo creo que es mejor `koalas`, que tiene la sintaxis de pandas que uno ya sabe. No es necesario aprender nada nuevo.

- **sparklyr** ((R), Pr: 2): La única vez que tuve que trabajar con data en Spark fue en Python y usé koalas. Pero vale la pena mencionar esta librería porque básicamente permite ejecutar Spark usando sintaxis de `dplyr`. Si es que llegaras a necesitar Spark, mi recomendación sería hazlo en otro lenguaje (principalmente por los problemas de memory leakage de R) pero si necesitas hacerlo en R, esta es la mejor opción.

- **NetworkX** (Rk: 147, ND: 20M+, Pr: 2): Es una librería de manipulación de datos, pero en forma de grafos. No la he usado más que para calcular métricas de centralidad (closeness, betweeness, degree, etc). Pero es probable que comience a utilizarla más.

- **Microsoft Excel** ((O), Pr: 1): Excel **nunca** debería ser una opción para trabajar con Datos, pero sí o sí tienes que saber usarlo porque lamentablemente los archivos `.xlsx` son todavía un formato extremadamente popular. <mark>NUNCA</mark> deberías utilizar Excel si no es sólo para entregar resultados. Si tú eres de los que aún dice que hay cosas que son más sencillas en Excel que en Pandas o SQL, es que no sabes utilizar bien esas tecnologías aún.

# Bases de Datos

- **sqlalchemy** (Rk: 49, ND: 49M+, Pr: 1): Esta es por lejos una de las mejores librerías que se han creado en Python. Básicamente permite utilizar cualquier Base de Dato SQL con una interfaz común. Debo decir que si bien esta es una librería extremadamente poderosa y que vale completamente la pena aprender, la documentación está pensada para gente bien "computín" y no es tan amigable. Mi recomendación para aprenderla es mediante videos tutoriales. Ahora en Ciencia de Datos la vas a ocupar sí o sí si eres Data Engineer para poder modelar Bases de Datos o hacer consultas. Como Data Scientist normalmente sólo la usarás como forma de conexión con Pandas mediante `create_engine` y `.to_sql()` para extraer datos.

- **sqlmodel** (Rk: 4085, ND: 90K, Pr: 2): Esta es una librería creada hace muy poco por el gran [Sebastián Ramírez](https://www.linkedin.com/in/tiangolo/?originalSubdomain=de) (Tiangolo). No he utilizado esta librería pero sí sé que está construida sobre sqlalchemy. `sqlmodel` es a `sqlalchemy` lo que `FastAPI` es a `Flask`. Por lo tanto, es muy posible que en el tiempo esta librería venga a reemplazar a SQLAlchemy principalmente porque Tiangolo dedica mucho tiempo a la buena documentación y casos de usos, cosa que SQLAlchemy no tiene tan bien hecho en mi opinión.

- **DBI** ((R), Pr: 1): DBI viene a ser una interfaz común para poder consultar datos. Creo que podría considerarse el símil de sqlalchemy, pero no sé si tiene tantas funcionalidades. Al menos esta siempre fue mi opción para conectarme a DBs en R, pero nunca me tocó modelar una base de datos como sí tuve que hacerlo en Python. DBI tiene conexión con casi todos los motores de SQL o usando conexión `odbc`.

- **PyMongo** (Rk: 185, ND: 16.8M+, Pr: 2): Esta es la interfaz para utilizar MongoDB desde Python. MongoDB es probblemente la base de datos no relacional más famosa. Sólo vale la pena si es que te toca trabajar con MongoDB pero lo bueno es que su uso es sumamente intuitivo. Utiliza la misma sintaxis que MongoDB pero en vez de usar el formato BSON (que es como un tipo de JSON), lo hace en los diccionarios de Python. Y por cierto, hacer queries en MongoDB es básicamente SQL con otra sintaxis y permitiendo data no estructurada como output, por lo que aprenderla es bastante sencillo.

- **elasticsearch-dsl** (Rk: 732, ND: 2.4M+, Pr: 2): Este no es la librería más popular para conectarse a ElasticSearch, que es un motor de base de datos basado en documentos que es extremadamente rápida. La sintaxis en ElasticSearch es horrible, y yo reconozco que no tengo idea como extraer datos usando ElasticSearch puro. El tema es que elasticsearch-dsl es tan intuitivo que pude generar procesos de ETL en ElasticSearch utilizando esta librería, ya que su API es como estilo dplyr (aunque es una librería de Python), lo que le permite ser muy expresiva y fácil de crear, leer y entender. Si alguna vez tienes que trabajar con ElasticSearch, usa esta librería ya que es muchísimo más sencilla.

- **psycopg2** (Rk: 194 y 99, ND: 31M + 15M, Pr: 0): El Ranking de esta librería es un poco extraño, la razón es porque si utilizas Windows descargas psycopg2, pero si tienes Mac o Linux descargas psycopg2-binary, por lo que en estricto rigor esta librería es la suma de ambos. Este es el cliente de Postgresql en Python, un motor de base de datos extremadamente popular y poderoso. Es una interfaz muy parecida a DBI en R. Es un cliente lowlevel y bien rápido para poder interactuar con DBs Postgres. Yo la he utilizado como motor tanto para DBs Postgres Puras o para Datawarehouse como Redshift que están basadas en Postgres. Además se puede conectar con `sqlalchemy`, por lo que diría que no es necesario aprender mucho su sintaxis porque saber `sqlalchemy` ya hace la pega.

- **pyodbc** (Rk: 161, ND: 19M, Pr: 0): Es una librería que nos permite hacer conexiones ODBC. Esta librería la usé únicamente en Windows para conectarme con Teradata que es un motor de Base de Datos que suele ser utilizado en entornos de alta seguridad como Bancos o Retail (mi recomendación: no usen Teradata, funciona bien, es rápido y todo pero su documentación al no ser código abierto es pésima, por lo que cosas fáciles se pueden hacer pero encontrar cómo hacer algo fuera de la común es casi imposible. Se los dice alguien que lo usó por 5 años). Normalmente se utiliza una línea para conectarse y es compatible con `sqlalchemy`, por lo que no es necesario aprender mucho.

- **Neo4J** ((O), Pr: 2): Debo decir que este tipo de bases de Grafo cambió demasiado mi manera de ver el almacenamiento de datos. Creo, luego de pelear con hartos motores de datos no estructurado que, este es la manera más sencilla de interactuar con datos NoSQL. Entre sus grandes pro está el hecho de que su sintaxis es muy fácil de aprender (parecida a SQL, pero no igual), es rápido, y no requiere joins.

- **rasterio** (Rk: 1454, ND: 749K+, Pr: 0): Esta es una librería para trabajar con rasters. Rasters son las típicas imágenes donde cada píxel está representado como un valor en una matriz/tensor. En el caso de rasterio, tiene más utility functions para trabajar con imágenes satélitales pero en general se utiliza como complemento a otras librerías. Normalmente se utiliza una que otra función.

- **Xarray** (Rk: 1454, ND: 749K+, Pr: 0): No sé si saben pero antiguamente pandas (que deriva de PAnel DAta ), tenía data panel, que es son varias realizaciones en el tiempo de un DataFrame, o sea un Pandas de 3 dimensiones. Bueno eso hace un tiempo se quitó de pandas y si querías más de 3 dimensiones necesitabas Numpy. Bueno Xarray permite la data panel, 3 Dimensiones, pero con nombre del nombres de array. Es una extensión que permite por ejemplo trabajar mejor con Imágenes Multiespectrales (ya que queda capa tiene un significado: RGB, Infrarrojo Cercano, Infrarrojo Lejano, etc.) y normalmente se combinan para poder crear índices y falsos colores para destacar ciertos aspectos de la imágen. Es una librería súper específica, por lo que sólo será útil cuando necesites trabajar con este tipo de datos.

- **Geopandas** (Rk: 733, ND: 2.4M+, Pr: 2): Esta es una extensión de Pandas, que incluye dos cosas interesantes a mi gusto, el incorportar shapes: Puntos, Polígonos, etc. Y el hecho de tener joins espaciales. De esta manera puedes combinar datasets si es que comparten mismo espacio, por ejemplo: Tienes puntos (coordenadas) de casas en un csv y tienes polígonos de regiones en otro csv. Al hacer join espacial, unirá los registros de casas que están dentro del polígono región igual que un join. El tema es que hay varios tipos de join espaciales: dentro, que colinden, que se intersecten, etc. Excelente librería, y no muy dificil de aprender.

- **Scikit-Image** (Rk: 325, ND: 8.6M+, Pr: 0): Esta es una librería de manipulación de Imágenes, muy parecida a OpenCV. Yo la usé una sola vez para intentar reconstruir una foto que rompí por error. Bien intuitiva tiene muchas built-in functions para manipular imágenes.

- **Spacy** (Rk: 475, ND: 5.1M+, Pr: 0): Esta es una tremenda librería para lidiar con texto libre. Tiene modelos pre-entrenados muy buenos en muchos idiomas para llegar y utilizar. Yo la usé una sóla vez porque en Cenco teníamos info sucia de muchas empresas (y querían sacar promociones en la tarjeta, o algo así): "Hipermercados Lider", "Supermercado Lider", "Falabella" , "Tiendas Falabella". Entonces hicimos un Name Entity Recognition para encontrar nombres de potenciales Comercios donde compraba la gente para poder ofrecer descuentos al sacar la tarjeta Cencosud. Por ejemplo, ellos tenían descuentos en Cine, y nadie y usaba la tarjeta para ir al cine. Pero sí la usaban para Uber, entonces querían cambiar la estrategia a ofrecer no sé 10 lucas en Uber o algo así. Aprendí lo que necesitaba en una tarde porque su documentación es excelente.

- **DBeaver** ((O), Pr: 2): Esta es un cliente de bases Open Source gratis (aunque también tiene una versión pagada). Básicamente es un software que puedes descargar que te permite conectarte a cualquier Base de Datos SQL y muchos otros. Entre los motores disponibles están: Postgresql, MySQL, Hive, ElasticSearch, Redshift, Snowflake y Neo4J entre otros. Además, en la versión paga te permite conectarte a MongoDB. Es rápido, tiene posibilidad de tener los modelos ER de cada Esquema además de varios atajos de teclado. Muy buena opción para conectarse con distintos motores.

# Visualizaciones

Esta es probablemente mi parte más débil principalmente porque es un área que no me gusta. Aún así he usado varias librerías, las cuales voy a mencionar ahora.

- **Seaborn** (Rk: 310, ND: 9M+, Pr: 1): Probablemente no esperaban que esta fuera mi primera opción. La razón por la que la menciono en primer lugar es porque es una librería con funcionalidades restringidas pero que hace la pega muy bien. Tiene la mayoría de gráficos prehechos y permite sin mucho código hacer gráficos muy bonitos y muy expresivos. Mi recomendación es sólo aprender `sns.catplot()` que permite graficar gráficos de variables categóricas o combinación categórica numérica (conteos, barplots, boxplot y amigos, etc.), `sns.relplot()` que permite generar gráficos para variables sólo númericas (scatter, lineplots) y `sns.displot()` que grafica básicamente histogramas. Estas 3 funciones tienen interfaz comunes con built-in facet y varias manera de agrupación (columnas, filas, colores, estilos, etc.). Una de las cosas que más me entusiasma es que Seaborn comenzó a desarrollar una interfaz muy similar a `ggplot2` de R lo cual la haría extremadamente flexible y fácil de usar. Definitivamente vale la pena aprenderla.

- **Matplotlib** (Rk: 110, ND: 26.9M+, Pr: 1): Yo creo que el ranking es un poco mentiroso, principalmente porque matplolib es dependencia de casi todas las librerías de gráficos, por lo que siempre la vas a necesitar. Lamentablemente hay que aprenderla. Y digo lamentable, porque a pesar de ser muy poderosa, considero que la documentación es como engorrosa y tiene una sintaxis muy verbosa. Además `seaborn` está construida sobre `matplotlib`, por lo que en casos de querer cambiar elementos del layout en `seaborn` se debe hacer mediante comandos `matplotlib`. Mi recomendación es aprenderla con ejemplos y algún cursito corto en Datacamp, porque es realmente difícil de aprender (no por su sintaxis sino que porque tiene muchas maneras distintas de hacer lo mismo y que a veces aplican y otras veces no). Igual me he dado cuenta que la termino usando más que Seaborn.

- **ggplot2** ((R), Pr: 1): Para muchos es la mejor librería de visualizaciones que existe. Y quizás tienen razón. `ggplot2` es un remake de ggplot (que fue un fracaso) y que está basado en el grammar of graphics que es un concepto en el cual las partes del gráfico se construye en capas (la figura; ejes; elementos como puntos, líneas, boxplots; cálculos como regresiones lineales, promedios, intervalos de confianza; etc.) Además como que por defecto la paleta de colores y los ejes son bien bonitos. Yo considero que no es tan fácil de aprenderla pero es la mejor sintaxis para graficar.
Existen algunas librerías/addins en RStudio como `esquisse` que permiten crear ggplots (te entrega el código incluso) con una interfaz tipo Tableau. Muy recomendada si trabajas en R y/o en Python. Además tiene un enorme ecosistema de librerías complementarias para poder graficar casi cualquier cosa.

- **plotnine** (Rk: 2473, ND: 232K+, Pr: 0): Es la versión en Python de ggplot2. Creo que es un tremendo esfuerzo y casi todas las funcionalidades están implementadas pero no funciona tan bien como ggplot2 (su ranking lo indica). El problema es que ggplot2 tiene muchos paquetes que lo complementan. Uno de los más poderosos es `patchwork` que es una interfaz para crear gráficos sobre gráficos de manera muy sencilla. Este es precisamente uno de las grandes problemáticas de plotnine, si se quieren crear layouts un poco más complejos comenzamos nuevamente a depender de `matplotlib` lo que evita una sintaxis única. Gracias a ver visto un EDA por [Martin Henze](https://www.linkedin.com/in/martin-henze/) utilizando ggplot comencé a usar esta librería pensando que podría lograr los mismos resultados, pero lamentablemente ggplot es `muy superior`.

{% include alert info='En mi opinión el 90% del tiempo utilizar gráficos estáticos será más que suficiente tanto para compartirlos en un PPT o para hacer EDAs. En caso de crear alguna aplicación interactiva entonces gráficos dinámicos e interactivos como los que hacen las siguientes librerías son una buena opción.'%}

- **plotly** (Rk: 359, ND: 7.5M+, Pr: 0): Plotly es una librería basada en D3, que a su vez es una librería de Javascript que se hizo muy popular gracias a su capacidad de desarrollar gráficos interactivos muy bonitos. Hoy tiene APIs en casi todos los lenguajes más populares. Para mí gusto es una librería que sólo vale la pena aprender si es que estás completamente dedicado a las visualizaciones. Si bien es una librería poderosa es muy verbosa. Afortundamente paquetes como `plotly-express` han aparecido para abstraer la verbosidad y crear versiones de gráficos comúnmente usados en pocas líneas.

- **plotly-express** (Rk: 2936, ND: 181K+, Pr: 2): Es la versión menos verbosa de plotly, si bien es un pelín menos poderosa debido a que es más simple, la mayor parte del tiempo será maś que suficiente. No entiendo por qué no es tan popular aún.

- **altair** (Rk: 360, ND: 7.4M+, Pr: 0): Es otra librería muy parecida a Seaborn en términos de sintaxis pero con la interactividad de plotly. Yo la utilicé sólo una vez creando una app en Streamlit. La razón: no quería usar plotly (en ese tiempo no conocía plotly express) y quedaban los gráficos más bonitos que en matplotlib y seaborn que eran estáticos. No vale la pena aprenderla y rara vez la verán por ahí.

- **bokeh** (Rk: 674, ND: 1.8M+, Pr: 0): Es otra librería proveniente de Javascript que puede ser usadas desde R o Python. La verdad es que no la he usado, pero pueden ser alternativas para plotly ya que también son interactivas basadas en HTML pero con una sintaxis más simple. Nuevamente las recomiendo sólo en caso de dedicarse el BI o al Data Storytelling donde vale la pena invertir en visualizaciones llamativas.

### Otras herramientas BI

- **Tableau** ((O), Pr: 2): En el caso de trabajar en Business Intelligence donde el foco es más mostrar herramientas interactivas que puedan manipular la data con algunos clicks, aparecen herramientas que no están basadas en código. Tableau es una muy buena alternativa. Es rápido, fácil de crear Dashboard con gráficos que sirven como filtros y pueden interactuar entre ellos. El problema, es que su costo es prohibitivo, su licencia es extremadamente cara y hoy existen otras herramientas más baratas que hacen lo mismo.

- **PowerBI** ((O), Pr: 2): Es el Tableau de Microsoft. Es una buena alternativa con costos de licencias bastante más bajo. Sigue la misma idea de Tableau de usar cajitas tipo Pivot Tables para crear gráficos. Igual de eficiente que Tableau pero mucho más barato.

- **Qliksense** ((O), Pr: 2): No recuerdo quien creó esto, pero es otra versión. Funciona exactamente igual que los otros dos. Tienen las mismas funcionalidades. Ninguna ventaja ni desventaja con los otros. 

{% include alert tip='¿Cuál elegir? Da lo mismo, es lo que tu empresa esté dispuesta a pagar.'%}

- **Shiny** ((R), Pr: 1): Podríamos decir que es la versión en R de estos productos. La diferencia es que es gratis, y es basado completamente en código. Permite crear todo tipo de Dashboards interactivos mezclando cualquier otra librería de R (aunque también se podría agregar Python mediante `reticulate`) tanto para manipular datos como para visualizar. Es extremadamente poderosa y flexible y hay varias empresas que crean sus portales utilizando Shiny. El problema es que no es tan fácil de hostear. En mi tiempo sólo RStudio ofrecía servicios para hostear ShinyApps (algunos gratis y otros de pago). Lo bueno es que se comenzó a crear todo un ecosistema en torno a `Shiny`, el cual tiene temas (basados en Bootstrap, material y otros frameworks de HTML, CSS y Javascript). Además, hay una librería llamada `golem`, que permite modularizar grandes aplicaciones e incluso se permiten ingresar elementos nativos en HTML, CSS o Javascript. Vale completamente la pena aprenderlo <mark>si es que</mark> te dedicas al BI en R y tienes tiempo de crear todo desde cero. Va a ser más flexible que Tableau, PowerBI o Qliksense, pero hay que crear todo.

- **streamlit** (Rk: 1361, ND: 853K+, Pr: 1): Similar a Shiny pero en Python. En mi opinión es mucho más sencillo de utilizar, pero mucho más simplista. Tiene lo justo y necesario para hacer funcionar una excelente aplicación demo. Lo bueno es que Streamlit fue comprado por HuggingFace por lo que se ha estado llevando sus funcionalidades para que sea el front-end de modelos de Machine Learning. Una ventaja de streamlit es que es fácilmente hosteable en cualquier servidor con Python (que son casi todos), en Heroku, en un servicio provisto por la misma gente de Streamlit o en Huggingface Spaces, siendo estos últimos totalmente gratis. En el caso de querer hacer una demo, se puede crear algo de gran calidad y complejidad en no más de una hora. Su sintaxis es muy sencilla y se puede aprender en unas horas.

- **Dash** (Rk: 1380, ND: 830K+, Pr: 0): Este es casi idéntico a Shiny (pero también en Python). Yo lo usé sólo una vez en un proyecto, y no nos gustó porque era muy complicado de setear. Básicamente crear el CSS que dejara los distintos divs en orden fue un martirio por lo que siempre nos quedaba la aplicación descuadrada. No vale la pena, ya que streamlit simplificó esto infinitamente.

- **Gradio** (Rk: 3187, ND: 148K+, Pr: 2): Es una interfaz aún más simple que Streamlit, pero con muchas menos funcionalidades. Esta librería sí que se creó con el sólo propósito de ser un IO para modelos de Machine Learning. A diferencia de Streamlit que puedes crear Dashboards, sitios webs, agregar gadgets, Gradio sólo le interesa crear gadgets de input/output para un modelo a modo de demo. Yo lo probé rápidamente y lo encontré muy fácil. Decidí aprenderlo luego de ver una demo de un Pipeline de Transformers por Omar Sanseviero, donde construyó un front-end con modelos de Generación de Texto y Machine Translation en 10 mins. Puedes ver su presentación [acá](https://www.youtube.com/watch?v=Mg7YeWBUKbM). Vale mencionar que también fue adquirido por HuggingFace por lo que puedes hostearlo facilmente en servidores Python, Heroku o Spaces. La gran ventaja de Gradio es que permite hostear de manera gratuita desde cualquier computador por dos días. Una vez se acabe puedes volver a levantar el servicio, el cual permite el frontend y una API en FastAPI creada automáticamente.

- **Django** (Rk: 357, ND: 7.5M+, Pr: 0): No lo he usado. Pero es por lejos la librería más poderosa de desarrollo Web. Acá ya no hablamos sólo de una interfaz de Dashboards sino que un software completo. Es tanto así que existen Ingenieros de Software especializados sólamente en el Ecosistema Django. Por nada del mundo como Data Scientist debieras tener que llegar a usar una librería tan poderosa como esta. Pero si te interesa crear una aplicación a nivel profesional con procesos de datos o Modelos de Machine Learning por abajo, esta podría ser una opción. Algunas aplicaciones creadas en Django son Instagram, Spotify, Youtube, Dropbox, entre otras.

- **Flask** (Rk: 88, ND: 35.9M+, Pr: 0): Tampoco lo he usado, pero tengo entendido que es un Django pequeñito, que además tiene otras funcionalidades como crear APIs. Es aún extremadamente popular en entornos de desarrollo web, pero en mi opinión está poco a poco cayendo en desuso, principalmente debido a que FastAPI está ganando mucho protagonismo en cuánto a APIs se refiere y es una opción mucho más sencilla de aprender.

# Machine Learning

Esta es por lejos mi sección favorita, por lo que puede que me extienda un poco más de que el resto.

- **Scikit-Learn** (Rk: 94, ND: 32.6M+, Pr: 1): Es la librería por excelencia para crear modelos de Machine Learning. La sintaxis de su API está tan bien diseñada que una manera de reconocer que otras librerías de Machine Learning son confiables es si es que siguen su API. Básicamente `scikit-learn` es super reconocida por sus modelos como Clase y su estandar `fit-transform-predict`, además de casi 15 años de vida. Si quieres hacer modelos de Machine Learning sí o sí tienes que partir por acá por varias razones: (1) Su documentación es excelente, incluso puedes aprender la teoría detras de cada modelo leyendo su [User Guide](https://scikit-learn.org/stable/user_guide.html) (toda persona que se dedique al ML debería leer la documentación completa de Sklearn una vez al año 🤪). Además contiene sólo modelos ML que están en el estado del arte. De hecho para que un modelo se implemente en Scikit Learn tiene que cumplir [requisitos](https://scikit-learn.org/stable/faq.html) muy estrictos. Andreas Mueller, mantenedor de Scikit-Learn tiene un curso disponible de manera gratuita [acá](https://www.youtube.com/watch?v=d79mzijMAw0&list=PL_pVmAaAnxIRnSw6wiCpSvshFyCREZmlM). Este es por lejos una de las mejores inversiones que uno hará como Data Scientist, ya que aprendiendo a utilizar esta librería podrás utilizar millones de otras basadas en la misma API. [Acá]({{ site.baseurl }}/titanic/)un ejemplo de modelamiento en Scikit-Learn.

- **tidymodels** (R, Pr: 2): Yo solía ser un fan de esta compilación de librerías. Creo que Max Kuhn es un tremendo desarrollador y lo respeto profundamente, pero creo que parsnip trató de llevar el modelamiento en R a un estado incluso más flexible que `scikit-learn` pero no les funcionó. Lamentablemente el Machine Learning en R está disgregado en muchas librerías todas con APIs diferentes, por lo que este esfuerzo de unificar todo es increíble. Lamentablemente el memory leakage que sufre R y el tremendo trabajo de los mantenedores de `scikit-learn` hacen que un esfuerzo como este nunca logre la popularidad que tiene Python en este rubro. Tidymodels está basado en 3 paquetes principalmente: `recipes`, para el preprocesamiento, que a mi gusto tiene una API muy similar a los Pipelines de Scikit, `parsnip`, que es la unificación de todos los modelos de ML implementados en R y `yardstick` que contiene todas las métricas de evaluación. Si te dedicas a hacer modelos pequeñitos de prueba, sin mucho intensidad de cómputo es una opción, en cualquier otro caso vale más cambiarse a `scikit-learn`.

- **caret** (R, Pr: 2): Este es el predecesor de `tidymodels`. A pesar de ser una librería que se le quitó mantenimiento hace un tiempo sigue disfruntando de mucha popularidad ya que tiene más de 200 modelos implementados. El propósito de Caret es el mismo de tidymodels sólo que su API no era compatible con el tidyverse por lo que decidieron seguir el esfuerzo de tidymodels. Este proyecto contaba con todo integrado, preprocesamiento, entrenamiento, postprocesamiento, esquemas de validación, métricas de evaluación, incluso ensambles. Por alguna razón lamentablemente decidieron cortarlo.

- **pycaret** (Rk: 1940, ND: 432K+, Pr: 2): Este es un proyecto en Python que nace de la base de Caret y que se ha hecho extremadamente popular. En mi opinión sólo vale la pena aprenderlo si es que no te gusta codear. Las ventajas es que permite hacer mucho en pocas líneas de código y es compatible con muchas librerías externas como XGBoost, LightGBM, etc. Además cuando uno no es experto en tareas menos habituales como Anomaly Detection o Series de Tiempo permite seguir el mismo esquema de código. Lo que me gusta del creador de esta librería es que él deja muy en claro que su objetivo que es que los Citizen Data Scientist pueden tener modelos de alta calidad a la mano. Creo que están haciendo un tremendo trabajo y he visto muchos Notebooks en Kaggle que lo usan y obtienen muy buenos resultados.

- **Feature Engine** (Rk: 3096, ND: 99K+, Pr: 1): Para mí esta es una librería de primerísima calidad. Tiene muy buen mantenimiento y tiene muchísimos mejores preprocesamiento que Scikit-Learn y además implementados en DataFrames. Contiene muchos de los excelentes encoders que tenía Category Encoders y además un Wrapper que permite convertir los preprocesadores de Scikit para que devuelvan pandas DataFrames en vez de Numpy Arrays. Espero que gane más popularidad, yo al menos la uso mucho.

- **category-encoders** (Rk: 814, ND: 2M+, Pr: 0): Esta solía ser mi librería de encoders por defecto, pero dejó de mantenerse porque los mantenedores se cansaron. En su momento fue muy buena y todavía tiene mucha popularidad. Particularmente encontré un par de issues que reporté pero se demoraron casi un año en corregirlo. Una pena.

- **statsmodels** (Rk: 294, ND: 9.6M+, Pr: 2): Si trabajas en Estadística en Python esta es la librería. Yo no soy muy fan de los modelos estadísticos, pero igualmente creo que es una librería interesante, porque también contiene muchas herramientas para trabajar con series de tiempo. En caso de necesitar mucho poder estadístico, creo que R es mucho más potente acá.

- **XGBoost** (Rk: 320, ND: 8.8M+, Pr: 1): Uno de los problemas que Scikit-Learn solía tener es que no tenía una buena implementación de algoritmos de Gradient Boosting (hoy tiene una buena implementación de HistGradientBoosting similar a LightGBM) y XGBoost quizás es la implementación más famosa que hay. Desde el 2014 viene dominando por lejos el modelamiento en data tabular y definitivamente es un algoritmo que hay que dominar. Si bien es cierto su performance es superior, llegar a esa performance es difícil de lograr, ya que hay que hacer un buen afinamiento de Hiperpárámetros. Definitivamente un algoritmo que hay que aprender.

- **LightGBM** (Rk: 393, ND: 6.5M+, Pr: 1): Me llama la atención que tenga menos descargas. Porque LightGBM para mí supera a XGBoost, por poco pero lo supera. En general para todas las competencias en la que he estado y modelos en producción que he dejado siempre obtengo mejor performance con LightGBM. Esta es una implementación liberada por Microsoft en 2016, y en mi opinión es bastante más rápido que XGBoost y menos complicado de afinar Hiperparámetros. El problema es la instalación, las docs de instalación son malitas, y la versión con GPU es bien enredada de instalar. Definitivamente, hay que tenerlo en el arsenal.

- **CatBoost** (Rk: 747, ND: 2.3M+, Pr: 1): Otro Gradient Boosting que está muy de moda. En mi opinión es el algoritmo más fácil de afinar. Casi no hay que mover los Hiperparámetros para obtener muy buenos resultados. Es fácil de instalar, pero en velocidad es similar a XGBoost. Creo que el único problema que le he visto es que cuando guardas el modelo es muy pesado. Por ejemplo, una vez entrené los 3 Boosting (típico en Kaggle) y no sé, XGBoost y LightGBM pesaban del orden de megas mientras que CatBoost pesaba 11 GB, no sé si habré hecho algo mal, pero encontré que era muy pesado. El otro contra (no tan contra), es que siempre queda fuera de los frameworks, y la API es un poquito diferente a Scikit. (XGBoost y LightGBM tienen versiones con API de Scikit). Definitivamente hay que aprenderlo.

{% include alert info='Lo bueno de los 3 grandes Boosting es que todos tienen Early Stopping y permiten el uso de un set de Validación mietras se entrena, igual que los algoritmos de Deep Learning.'%}

- **DeepChecks** (Rk: NA, ND: NA, Pr: 2): Yo no lo he usado aún en mis pegas, pero he hecho pruebas y revisado a fondo la documentación y creo que es una excelente librería para estudios previos de la data (chequear potenciales drifts y el potencial poder de generalización de un modelo) y para monitoreo. Permite realizar distintas validaciones para entre tu set de entrenamiento y tu data real, o test set para chequear que el modelo funciones bien en el tiempo.

- **Mapie** (Rk: NA, ND: NA, Pr: 2): Excelente librería para aplicar Conformal Prediction, es decir, se pueden generar predicciones con intervalos de confianza en Regresión y Clasificación probabilística para modelos de clasificación. Lo bueno es que es solo un wrapper y es Scikit-Learn compatible. Tuve la oportunidad de estudiar la documentación a fondo y es realmente la manera de generar modelos robustos en especial cuando hay mucha incertidumbre.

- **mlxtend** (Rk: 1024, ND: 1.4M+, Pr: 2): Tremenda librería creada por Sebastian Raschka, profesor de Wisconsin Madison y parte de Lightning AI. Es un complemento a Scikit-Learn y tiene varios elementos que permiten extender las capacidad de Scikit. En particular rescato las herramientas para ensambles tipo Stacking. Muy necesaria si quieres competir, y si quieres un modelo ensamblado.

- **pyGAM** (Rk: 2237, ND: 325K+, Pr: 0): Es una librería que hace modelos GAM (Generalized Additive Models). Estos modelos son famosos por ser la mejor mezcla entre buena predicción y buena explicabilidad. Quizás el modelo GAM más conocido es `prophet` de Meta. En general esta librería no me gustó, y si es que realmente quieres meterte en este tipo de modelos mejor utilizar `mgcv` en R que es años luz más maduro. No creo que valga la pena aprenderlo.

- **CuML** (Rk: NA, ND: NA, Pr: 0): Esta es una librería que está aún en desarrollo por parte de NVIDIA, pero es la parte de ML de cuDF y cuPY. Es un mirror de Scikit-Learn, pero que corre en GPU. En especial algoritmos como Random Forest y SVM pueden verse muy beneficiados. No creo que valga la pena aprenderlo, porque es lo mismo que Scikit-Learn.

- **Imbalanced-Learn** (Rk: 650, ND: 3M+, Pr: 0): Es la librería por excelencia para desbalance de clases. Lo bueno es que incluye técnicas de undersampling, oversampling, SMOTE y algoritmos propios que funcionan con desbalance como RUSBoost y BalancedRandomForest. Debo confesar que casi nunca obtengo mejores modelos utilizando estas estrategias, y no me ha tocado usarlo aún, pero normalmente utilizando el parámetro sample_weigths de cualquier modelo de Scikit-learn podría funcionar mejor.

- **Shap** (Rk: 530, ND: 4.1M+, Pr: 1): Es hoy quizás la librería más poderosa para dar explicabilidad. Existen varios spin-offs enfocados en problemas específicos pero creo que es algo que todos deberíamos dominar porque al negocio siempre le interesa entender por qué un modelo predice lo que predice.

- **ELI5** (Rk: 1022, ND: 1.4M+, Pr: 2): Otra opción para la explicabilidad de modelos. No lo he usado pero solía ser la librería por defecto antes que apareciera el boom de los shap values.

- **Implicit** (Rk: 2761, ND: 207K+, Pr: 2): Librería de Factorization Machines para modelos de recomendación Implicita. Esta la usé una vez para una prueba de Concepto en Cencosud. Fácil de usar, buenos tutoriales, me gustó. No tengo más que decir, porque fue "el uso" que le dí.

- **Surprise** (Scikit-Surprise) (Rk: 2860, ND: 195K+, Pr: 2): No alcancé a usarla, porque en la misma Prueba de Concepto anterior me dí cuenta que teníamos un recomendador implícito y Surprise es para modelos explícitos. Para tenerlo en cuenta. 

- **LightFM** (Rk: 1920, ND: 441K+, Pr: 2): Esta fue la librería que terminé utilizando, debido a su rápidez. Recuerdo que en ese momento no pude sacarle todo el potencial porque funciona mejor en entornos Unix y obvio, nos obligaban a usar Windows. También para tenerla en cuenta.

- **H2O** (Rk: 1954, ND: 428K+, Pr: 2, (R)): Es una librería que está tanto en Python como en R que por detrás corre una JVM. Es la librería en CPU más rápida que he visto. Yo sólo la ví en curso en R con Erin Ledell. Es buena para hacer cosas rápido. Además posee AutoML y Stacking, para los que les guste algo rápido con poquito código.

- **Prophet** (Rk: 1367, ND: 848K+, Pr: 2): Hace poco hubo un escándolo porque la empresa Zillow hizo un uso indiscriminado de Prophet entrenando modelos sin entender y eso le significó un impacto muy negativo (pueden leer más al respeco [acá](https://towardsdatascience.com/in-defense-of-zillows-besieged-data-scientists-e4c4f1cece3c)). Pero si se le da un uso correcto, creo que es una tremenda librería. Es fácil de usar y tienen muchas ventajas. Konrad Banachewicz está haciendo un curso de series de tiempo en el canal de [Abishek Thakur](https://www.youtube.com/channel/UCBPRJjIWfyNG4X-CRbnv78A) y habló sobre este modelo, y la verdad lo encontré muy interesante. Úselo con precaución y bajo su propio riesgo.

- **Neuralprophet** (Rk: 4635, ND: 68K+, Pr: 2): Spin-off de Prophet pero utilizando algoritmos de Redes Neuronales. Mismo cuidado que con prophet.

- **Sktime** (Rk: 2739, ND: 211K+, Pr: 2): Es una extensión de Scikit-Learn para modelos aplicados a Series de Tiempo. Tiene algoritmos propios para clasificación (de series de tiempos, o sea clasificar un secuencia), regresión, forecast (no es lo mismo que regresión), anomaly detection y tiene varios CV propios de series de tiempo. Yo no la usé propiamente tal, pero aprendí mucho leyendo su documentación, en especial para entender la diferencia entre forecast y regresión. Además posee un transformer que permite convertir modelos de forecasting en Regresión. Muy buena librería si trabajas con series de tiempo.

- **Skforecast** (Rk: NA, ND: NA, Pr: 2): Muy similar a sktime pero creada por Joaquin Amat, un data scientist español. Creo que siempre el trabajo en español tiene que ser destacado.

- **TSFresh** (Rk: 1888, ND: 456K+, Pr: 2): Yo utilicé esta librería como herramienta de feature extraction para series de tiempo. Posee una función `extract_features` que permite crear muchísimas features para series de tiempo. Muy buena librería.

- **Lifetimes** (Rk: 1473, ND: 731K+, Pr: 2): Librería especializada en Survival Models. Los modelos de sobrevivencia son modelos que buscan estimar el tiempo a un evento. Lo utilicé en la competencia de Mercado Libre, pero no me dió muy bueno así que seguí por otro lado. Es bueno tenerlo como alternativa para tipos de modelación no tan comunes.

- **Boruta-Shap** (Rk: NA, ND: NA, Pr: 0): Es una librería muy pequeñita que permite utilizar el algoritmo Boruta más Shap Values para Feature Selection. Por defecto utiliza un Random Forest para escoger las variables más importantes, pero yo lo utilicé con XGBoost y LightGBM en GPU y funciona bastante bien.

- **LOFO** (Rk: NA, ND: NA, Pr: 0): Es otra librería de Feature Selection. En este caso la ventaja que ofrece sobre Boruta Shap es que se realiza una selección utilizando un modelo específico pero en un esquema de Cross Validation. Esta la utilicé en una competencia que tenía muchas variables anónimas, y funcionó bastante bien.

- **Optuna** (Rk: 613, ND: 3.2M+, Pr: 1): Es probablemente la mejor librería de optimización que hay hoy. Originalmente permite resolución de algoritmos de Optimización (min, max, minmax). Pero su gran fortaleza es que permite la implementación de algoritmos Bayesianos de búsqueda compatibles con modelos de Machine Learning y Deep Learning (agregando Pruning, que permite terminar la búsqueda en espacios poco prometedores). Es obligación aprender a utilizarla, 100-200 iteraciones de Optuna es equivalente a una búsqueda gigantesca en GridSearch o RandomSearch.

- **Scikit-Optimize** (Rk: 613, ND: 3.2M+, Pr: 0): No lo he usado, pero es competidor directo de Optuna. No sé mucho más pero creo que era necesario mencionarlo.

- **Hyperopt** (Rk: 809, ND: 3.2M+, Pr: 0): Idem al anterior.

- **Scikit-plot** (Rk: 1756, ND: 520K+, Pr: 0): Es en estricto rigor una librería de visualizaciones, pero sólo tiene visualizaciones asociadas a Machine Learning. Es muy sencillo de usar y con un comando permite graficar matrices de confusión, curvas ROC, curvas Precision-Recall, Curvas de Aprendizaje, Silhouette, Curvas de Calibración, etc. Yo comencé utilizandola porque antes los Plot de Scikit-Learn quedaban muy feos. Esto está muy mejorado actualmente y recomendaría utilizar este tipo de librerías sólo para curvas muy específicas.

- **Yellowbrick** (Rk: 1659, ND: 578K+, Pr: 0): Para mí, hace y tiene exactamente lo mismo que Scikit-Plot. No recuerdo por qué comencé a usar Scikit-Plot por sobre esta.

# Deep Learning

- **Pytorch** (Rk: NA, ND: NA, Pr: 1): Es el framework de Deep Learning de Meta. Quizás esto es sorpresivo. Pero la razón por la que Pytorch no está en el Ranking es porque se recomienda su instalación via Conda. Para mí (y esto es muy sesgado), es la mejor librería de Deep Learning. Y la razón es porque te permite entender el funcionamiento de una red neuronal de mejor manera que con otros frameworks. El contra de Pytorch es que necesitas mucho código para entrenar principalmente, pero permite entender muy bien cuando hay que setear los gradientes a cero, en qué parte se evalúa la loss function, cuando haces backpropagation y actualizas los pesos. Además como te fuerza a utilizar clases permite mejorar tu programación orientada a objetos y su gran fuerte es la documentación, muy buena en términos de uso, pero también de teoría. Otro aspecto espectacular de Pytorch es que permite el desarrollo de spin-offs que mencionaré más tarde. ¿Es Pytorch perfecto? la verdad es que no. Como dije antes, es muy verboso y entrenar en Aceleradores es engorroso. Hay que estar consciente en todo momento de si tu tensor vive en CPU o GPU, hay que moverlo manualmente. No, es un cacho. Aún así, creo que es necesario hacer al menos un par de modelos en Pytorch Nativo, acá un [ejemplo]({{ site.baseurl }}/pytorch-native/). Si quieres iniciarte en Pytorch, lo mejor es partir por el 60 minutes [Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html).

- **Pytorch-Lightning** (Rk: 692, ND: 2.7M+, Pr: 1): Pero afortunadamente existe Pytorch Lightning que soluciona todos los inconvenientes de Pytorch Nativo. Permite organizar mucho del excesivo código de Pytorch y tiene una API que permite escalar a GPUs, TPUs, IPUs y HPUs sin casi ningún cambio. Además permite la portabilidad del código, haciendo que un mismo módulo sea muy fácil de reutilizar casi sin latencia. Creo que definitivamente Lightning es la razón por la que me enamoré de Pytorch. Dentro de los mejores lugares para entender bien el funcionamiento de Pytorch Lightning está este el [level up](https://pytorch-lightning.readthedocs.io/en/latest/expertise_levels.html) y una serie de [tutoriales](https://pytorch-lightning.readthedocs.io/en/latest/notebooks/course_UvA-DL/01-introduction-to-pytorch.html) de la Universidad de Amsterdam y [otros más](https://pytorch-lightning.readthedocs.io/en/latest/tutorials.html).

- **Tensorflow** (Rk: 181, ND: 17.2M+, Pr: 0): Es el primer framework de Deep Learning liberado por Google en el 2015. Esto no es sesgo. No conozco a nadie que haga sus modelos utilizando Tensorflow. Inicialmente el hecho de tener ejecución estática, hacía que fuera muy difícil programar en él, además de que se sentía como programar en otro lenguaje distinto de Python. La versión dos permite ejecución dinámica, para debuggear en tiempo real, pero siento que ya quedó muy por detrás de Pytorch. Ahora, ¿por qué tiene tantas descargas? Porque se necesita el backend para programar en Keras que sí es la manera en que todo el mundo usa Tensorflow.

- **Keras** (Rk: 292, ND: 17.2M+, Pr: 1): Para los que no les gusta complicarse con Pytorch pero igualmente quieren utilizar Redes Neuronales, Keras es la solución. Es por lejos la API más famosa, y más sencilla de aprender. Es un poco más lento que Tensorflow puro pero se encuentran muchos tutoriales de cómo implementar modelos sencillos. Yo comencé a aprender redes neuronales en Keras pero me fui desalentando porque no me gustó la documentación, la encontré muy engorrosa, y además porque empecé a confundirme. Hay como 3 formas distintas de implementar modelos, hoy algunas muy parecidos a Pytorch utilizando clases. No hay mejor o peor entre Keras o Pytorch, pero Pytorch está ganando mucha popularidad, mientras Tensorflow la pierde. Acá tengo un pequeño [ejemplo]({{ site.baseurl }}/keras/) de cómo utilizar Keras.

- **Jax** (Rk: 1546, ND: 662K+, Pr: 0): No la he usado, y no está en mi agenda aprenderlo, pero puede que gane mucha popularidad. Corresponde a otro framework desarrollado por Google y fue adoptado por DeepMind, por lo que quizás debido al tremendo desarrollo que ellos hacen comience a hacerse famoso.

- **Pytorch-Geometric** (Rk: 4190, ND: 86K+, Pr: 2): Es una extensión de Pytorch para trabajar con Redes de Grafos (Geometric Deep Learning). Yo lo encontré difícil de aprender, pero no por el framework sino que las redes de Grafos son más enredadas. Para tenerlo en cuenta. Tiene muy buena [documentación](https://pytorch-geometric.readthedocs.io/en/latest/), por lo que pueden comenzar el aprendizaje por ahí.

- **Pytorch-Forecasting** (Rk: 4346, ND: 78K+, Pr: 2): Es un spin-off de Pytorch para Forecast utilizando Redes Neuronales. Lo interesantees que tiene varios algoritmos famosos implementados como N-Beats, DeepAR y Temporal Fusion Transformer. Además tiene Dataloaders que están diseñados para tipos de predicción propios de series de tiempo. Yo no la utilicé pero sí estudié bastante sus docs para ver si podía utilizarla.

- **Pycox** (Rk: NA, ND: NA, Pr: 2): Spin-Off de Pytorch para el uso de Modelos Survival en Deep Learning. Tampoco alcancé a utilizarla, pero también para tenerla en el arsenal si usamos este tipo de modelos.

- **torchvision** (Rk: 518, ND: 4.3M+, Pr: 0): Es una librería auxiliar a Pytorch para Visión que provee de datasets, Data Augmentation y algunos modelos preentrenados. Particularmente creo que hoy no vale la pena. Existen otras librerías más potentes que esta y se demora mucho en incluir cosas nuevas. No vale la pena a mi gusto.

- **Albumentations** (Rk: 2391, ND: 283K+, Pr: 1): Es por lejos la mejor librería de Data Augmentation en Imágenes. No sólo es rápida sino que permite augmentation de Imágenes y Masks. No es muy dificil de aprender y es compatible tanto con Pytorch como con Tensorflow/Keras. Muy buena librería.

- **Kornia** (Rk: 1918, ND: 441K+, Pr: 2): Si bien Albumentations funciona sumamente rápido, funciona en CPU. Kornia es un Albumentation en GPU, lo cual permitiría, es especial en multiple GPU, tener una que se dedique al preprocesamiento. No la he usado, pero está ganando mucha popularidad.

- **OpenCV** (Rk: 466, ND: 5.2M+, Pr: 0): Si bien es una librería agnóstica de Visión, posee algunos modelitos internos que funcionan súper bien de manera rápida para tareas de detección de objetos, segmentación, etc. con los que fácil y rápidamente puedes impresionar. Yo la uso principalmente junto con Albumentations y es espectacular. 

- **Timm** (Rk: 837, ND: 1.9M+, Pr: 0): Si te dedicas a la Visión Computacional tienes que conocer esta librería, debe tener un par de comandos y su principal función es descargar modelos pre-entrenados. Principalmente sus modelos son compatibles con Pytorch pero creo que ya se pueden utilizar en Tensorflow/Keras también. Lo mejor de esta librería es que en quizás un par de semanas de salido una arquitectura estado del arte (SOTA Model) ya va a estar disponible acá. Puedes encontrar desde MobileNet o ResNets, hasta ViT, ConvNext, EfficientNets, y un largo etc. Otra buena noticia es que Timm se asoció con HuggingFace por lo que muy probablemente será aún más rápido ver avances en arquitecturas ultra modernas.

- **Transformers** (Rk: 403, ND: 6.3M+, Pr: 1): Es quizás por lejos la librería que más rápido ha crecido en el último tiempo y es mantenida por HuggingFace. Inicialmente estaba enfocada en proveer modelos preentrenados y tokenizers de modelos de NLP. Hoy tiene modelos, de Visión, Audio, y dicen que vienen de Grafos. Yo no la he usado mucho, porque no estoy muy metido en el área de NLP, pero hay que conocerla. Con un par de líneas puedes hacer un tremendo transformer estado del arte. Aprovecho de destacar el trabajo que hace la Universidad de Chile que tiene un Bert preentrenado en español disponible para uso libre, el [Beto](https://github.com/dccuchile/beto).

- **torchinfo** (torch-summary) (Rk: 4030, ND: 93K+, Pr: 0): Es una librería pequeñita que con una función `summary` permite ver un detalle de la red neuronal: capas, parámetros, tamaño, peso, idéntico a como lo permite Keras. Es una función literalmente.

- **torchmetrics** (Rk: 705, ND: 2.6M+, Pr: 0): Es una excelente librería con métricas de evaluación para Deep Learning. ¿Por qué no usar las típicas de Scikit-Learn? Primero, esta tiene muchas más métricas específicas para NLP, Object Detection y un largo etc. Además estas métricas se pueden ejecutar en GPU o Clusters, dependiendo de la paralelización o distribución, lo cuál las hacen mucho más rápidas.

# Misceláneo

- **VSCode** ((O), Pr: 1): Para mí, el mejor IDE para programar hoy en día, aunque es agnóstico, puedes programar casi lo que quieras acá. Principalmente porque es liviano, evita muchas complejidades para trabajar en ambientes aislados (conda o venv). Lo bueno es que es totalmente personalizable, tiene extensiones para todo. De hecho este blog es escrito en Markdown utilizando distintas extensiones que me facilitan la escritura. Otro aspecto para los más computines es que puede utilizar keycodes populares como Vim, Emacs o Sublime para usar sólo el teclado. En particular VSCode tiene muy buen soporte para Python permitiendo el uso de Notebooks, Scripts o una Consola Interactiva. Además es posible utilizar terminal (aunque increíblemente no funciona tan bien en Windows, por eso Linux for the "Win"), tiene soporte de GIT, debugger y un largo etc. Vale la pena, toma un tiempo aprenderlo pero no se van a arrepentir.

- **RStudio** ((O), (R) Pr: 1): Hay que decir que este es por lejos el IDE más optimizado para R. Permite instalar librerías directo de CRAN, tiene visualizador de Datasets, un sector de Plots, Documentación incluida, Explorador de Archivos y Terminal. Además tiene integración con GIT y librerías como blogdown (para hacer tu sitio web en R, mi antiguo sitio fue hecho ahí), bookdown (mi tesis de pregrado la escribí ahí), etc.

- **Spyder** ((O), (R) Pr: 2): Este es como una réplica de Rstudio pero para Python. Inicialmente cuando me moví de Python comencé a utilizarlo, y es bien completo, permite Scripts, tiene extensiones para Notebooks, tiene explorador de variables. A mí particularmente me molestaban dos cosas, que se demora en iniciar, y que nunca pude encontrar una paleta de colores para el highlighting. Es una buena opción para programar en un ambiente que está diseñado para ciencia de datos.

- **Pycharm** ((O), (R) Pr: 0): También lo utilicé con licencia completa y debo decir que si bien es un IDE enfocado exclusivamente en Python me cargó. Siento que no está pensado para Ciencia de Datos. Es muy pesado, se demora mucho en partir, su configuración inicial es terrible y al menos a mí siempre se me quedó pegado. Jetbrains (los creadores de esto) creo que se dieron cuenta que no era lo mejor y crearon un IDE enfocado en Ciecia de datos ([DataSpell](https://www.jetbrains.com/es-es/dataspell/)), pero la verdad no lo he probado. Es tan completo que llega ser abrumante, y nunca pude aprender todo lo que podía eventualmente servirme. Para mí, no vale mucho la pena.

- **Atom** ((O), (R) Pr: 0): Para mí era lejos el mejor IDE para programar, creado por Github. Tiene extensiones, muy buenos atajos de teclado, era rápido, liviano y tenía una extensión llamada Hydrogen que permitía tener los resultados de tu código directamente en el Script de manera muy intuititva y cómoda. ¿Por qué dejé de usarlo? Siento que dejaron de darle tanto soporte luego que Github fue adquirido por Microsoft y favorecieron más VSCode. Además luego de usarlo por un rato, comandos simples como `df.shape` tomaba 40-50 segundos, lo cual era inaceptable. Créanme que volvería mil veces a utilizarlo si viera que hay soporte y mantenimiento continuo. Una lástima.

- **GIT/Github** ((O), Pr: 1): Me cuesta creer que aún existen muchos "Data Algo" que no usan GIT. Esto debería ser obligación y requisito siempre. Afortunadamente me tocó trabajar en un equipo con muy buenas prácticas de desarrollo donde entendí la importancia de llevar control de versiones siempre. GIT no es difícil, pero es importante entender conceptos de Commits, push, trabajo en ramas. Adicionalmente llevarlo con Github (u otras variantes como GitLab o BitBucket) y entender conceptos como Pull Request, levantar Issues, Revisiones de códigos, approvals, etc. Si no usas GIT/Github, no te sientas mal. Hay empresas gigantes que no lo usan, pero aprenderlo y fomentar su uso te lleva fácilmente a un nivel más alto de calidad. Si quieres aprenderlo tengo una serie de tutoriales que parten [acá]({{ site.baseurl }}/github/).

- **Docker** ((O), Pr: 1): Hoy por hoy es imprescindible mover a producción todo en Docker. No soy para nada experto en el tema pero puedo crear un contenedor, conectarlo con el mundo real y eventualmente hostearlo en alguna parte. Es por lejos la mejor manera de asegurar reproducibilidad en cualquier ambiente (Unix, Max o incluso Windows con WSL2). Hay que aprenderlo sí o sí.

- **Bash** ((O), Pr: 1): Creo que es sumamente importante conocer un poquito de Bash, en especial para automatizar procesos. Bash es el lenguaje de tu computador y te permite interactuar con él. Algunas cosas interesantes que puedes hacer: Agendar trabajos periódicos de manera automática, mandar correos cuando termine un proceso largo, apagar el computador luego de entrenar un modelo por la noche. No es dificil de aprender, y la mayoría de las veces vas a googlear en Stackoverflow para salir del paso.

- **Wandb** (Rk: 791, ND: 2.1M+, Pr: 2): Este es un logger, que si bien permite llevar registro de modelos de ML y DL, funciona mejor en Deep Learning. Fácil de usar, muy linda interfaz y permite llevar registro de Arquitectura, Hiperparámetros, Curvas de Aprendizaje, ejemplos de Inferencia, almacenar tablas y gráficas, etc. Además contiene un sistema de Búsqueda de Hiperparámetros distribuido usando Hyperband, es decir, se puede entrenar el mismo modelo en distintas máquinas sin interferir entre ellos y sin repetir búsqueda, Weights & Biases lleva el control.

- **MLFlow** (Rk: 260, ND: 11.1M+, Pr: 2): La verdad es que MLFlow es igual o mejor que Weights & Biases, pero a mí no me gustó. Encuentro que su documentación es engorrosa y su API no es tan intuitiva. Hace lo mismo además de poder llevar proyectos y un Model Registry para llevar control de versiones del entrenamiento de tu modelo. Si les interesa aprenderlo, tengo un tutorial [acá]({{ site.baseurl }}/mlflow/).

- **FastAPI** (Rk: 377, ND: 6.8M+, Pr: 1): Es quizás una de las librerías más rápidas en Python y es muy fácil de usar. Primero, está hecha por un Colombiano (Tiangolo), es de excelentísima cálidad, muy buena documentación, muchas funcionalidades, y requiere de poquito código, ¿qué más se puede pedir?. Definitivamente si quieres distribuir lo que sea, data, un modelo de ML, FastAPI es la mejor opción. Es tanto la popularidad que varias librerías utilizan esta librería under the hood.

- **Airflow** (Rk: 375, ND: 6.8M+, Pr: 1): Yo creo que a menos que seas Analista de Datos, es una herramienta que hay que aprender. Airflow es un orquestador creado por Airbnb, que permite ejecutar y agendar Scripts para ser ejecutados de manera local o remota. Lo bueno de Airflow es que servicios como AWS, o Astronomer permiten ejecutarlos en entornos autoescalables en Kubernetes, lo cual quita una capa de complejidad, en especial a los que no sabemos cómo demonios funciona Kubernetes (un orquestador de contenedores). Airflow se hizo famoso como un orquestador de ETLs, que es compatible con casi todo. Yo lo he usado con: AWS, Spark, AWS Glue, ElasticSearch, MongoDB, SQLAlchemy, Redshift, Postgres. Es tan potente que incluso permite entrenar modelos de ML localmente o en entornos como Amazon SageMaker (aunque no es la opción óptima para ML), de hecho Airbnb creó BigHead para eso, que no ha sido liberado al público. Creo que su único contra es que un poco verboso, y tiene harto código boilerplate. Pero su funcionamiento es impecable.

- **Metaflow** (Rk: NA, ND: NA, Pr: 2): Otro orquestador, pero creado por Netflix, pero que está enfocado en llevar modelos de ML a producción. Las ventajas, mucho menos boilerplate que Airflow, no tienes los típicos problemas de Xcoms en Airflow, puede ejecutarse local o en AWS mediante AWS Batch, EC2 y Step Functions. Permite automatizar todo el proceso de entrenamiento creando de ser necesarios ambientes anacondas independientes para cumplir con requerimientos de versiones específicas. No alcancé a utilizarlo, pero me tocó leerme toda la documentación para impulsar su uso.

- **Kedro** (Rk: 2066, ND: 378K+, Pr: 0): Otro orquestador, pero desarrollado por QuantumBlack. Es bien poderoso, en el sentido que permite crear Pipelines de carga de datos, y de entrenamiento de modelos, pero no logré encontrar tantas opciones de escalabilidad. Si bien permite por ejemplo conexión con Sagemaker en AWS, no tiene las opciones más avanzadas de escalamiento vertical y horizontal que tiene Airflow y Metaflow. Además lo encontré en su momento un poco verboso, y sus Docs tenían errores, que hizo que me costará mucho entenderlo. 

- **DVC** (Rk: 1700, ND: 551K+, Pr: 1): Para mí es el orquestador más liviano, con menos Boilerplate y más sencillo de utilizar, pero tiene una cierta inclinación al entrenamiento de modelos. DVC es más que un orquestador, permite llevar registro de versiones de tu data, los cuales normalmente no es posible llevar en GIT; organizar Pipelines, llevar registro de Hiperparámetros, guardar métricas de performance, etc. Me gustó mucho más que Airflow, pero para una orquestación local, aunque podría escalar. Puedes aprender de él en este [tutorial]({{ site.baseurl }}/dvc/).

- **Great-Expectations** (Rk: 429, ND: 5.9M+, Pr: 1): Es un validador de datos. No les puedo explicar lo necesario que es empezar a incluir elementos como estos en nuestros Pipelines de datos. Todas las empresas tienen datos, pero pocas empresas con calidad suficiente para llegar y utilizar. Great Expectations es como una librería de Tests asociados a si los datos cumplen: rangos, tipos, cantidad, distribución y un largo etc. En caso de no cumplir levanta la alerta dando en detalle qué registros no cumplen con el estándar solicitado. Además es compatible con Airflow, por lo que uno puede usar como Gate de Ejecución si tu data cumple o no los requerimientos de modo de no cargar datos sucios en tus fuentes principales de almacenamiento. Muy buena librería.

- **Pytest** (Rk: 72, ND: 39.4M+, Pr: 1): Librería de Unit Test, algo que los Data Scientist rara vez hacemos. Es muy buena librería, fácil de usar, aunque es media rara la Documentación, pero nada que un buen tutorial de Youtube no pueda enseñar. Todos los pipelines de datos, deberían considerar Unit Tests.

- **Hydra** (Rk: 1507, ND: 698K+, Pr: 1): Para mí el mejor CLI para modelos de Machine Learning, nacida en el equipo de Research de Facebook, ahora Meta. No sólo permite crear comandos personalizados para ejecutar Scripts desde el terminal sino que también permite crear configuraciones muy complejas tanto para modelos de ML como para Pipelines en general. Para los que siguen el Blog saben que es de mis favoritas, y pueden ver [ejemplos]({{ site.baseurl }}/hydra/) acá. Muy buena librería, aunque no es tan famosa aún.

- **CML** (Rk: NA, ND: NA, Pr: 0): Es una librería para automatizar procesos con Github Actions. Si les interesa ver en acción pueden chequear [acá]({{ site.baseurl }}/cml/). No vale la pena aprenderla, son sólo un par de comandos y ya.

- **BentoML** (Rk: NA, ND: NA, Pr: 0): Esta es una librería que permite automatizar el Deployment de Modelos de Machine Learning. No la he usado pero he leído mucho su documentación, porque en estricto rigor permite crear de manera muy sencilla un Docker con tu modelo que esté listo para entregar al equipo de desarrollo. También crea una API Rest automáticamente. Definitivamente voy a estar metiéndome más en el tema.

- **MLEM** (Rk: NA, ND: NA, Pr: 0): Esta es una librería que me ofrecí a probarla en Beta. Hace lo mismo que Bento, pero permite rápidamente deploy en Cloud (AWS, Azure y GCP y Heroku), para cualquier tipo de modelo, y crea el Docker automáticamente. Cuando la ví me pareció demasiado mágica y está recién partiendo. Lo bueno es que incluye un curso que se puede tomar de manera gratuita [acá](https://learn.iterative.ai/).

- **Typer** (Rk: 349, ND: 7.8M+, Pr: 2): Creada también por Tiangolo, es un CLI mucho más poderoso que Hydra pero con un enfoque general. Su API es muy parecida a FastAPI, muy sencilla y potente. Yo la probé antes de conocer Hydra, pero igual creo que vale mucho la pena.

- **BeautifulSoup4** (Rk: 63, ND: 42M+, Pr: 2): Es una herramienta de Scrapping para poder tomar data de páginas web. Súper potente, ya que tiene mucho del trabajo que uno normal necesita hacer automatizado. Su documentación es buena y es fácil de aprender. Si quieres saber cómo usarla tengo un tutorial [acá]({{ site.baseurl }}/dtc/).

- **Boto3** (Rk: 1, ND: 392M+, Pr: 0): Es impresionante la cantidad de descargas de Boto3. Lamentablemente sólo será útil si utilizas AWS. Yo la he usado principalmente para interactuar con S3. Además si instalas `s3fs` y `fsspec` es posible utilizar pd.read_* y .to_* de pandas utilizando un URI de S3 directamente, por ejemplo: `pd.read_csv('S3://bucket/folder/file.ext')`.

- **Joblib** (Rk: 126, ND: 24M+, Pr: 0): Yo lo uso principalmente para serializar modelos entrenados de Scikit-Learn y similares de acuerdo a [esto](https://scikit-learn.org/stable/model_persistence.html).

- **Pickle** (Rk: NA, ND: NA, Pr: 0): Dejé de usarlo, porque Scikit-Learn favorece guardarlo en formato joblib.

- **Faker** (Rk: 454, ND: 5.4M+, Pr: 0): Yo sólo lo utilicé para una prueba de técnica para un candidato. Quería poner la misma data que utilizabamos pero sin entregar información confidencial. Faker permite emular info de manera muy real, creando de todo. En ese momento, cree: Nombres, Apellidos, Direcciones, Teléfonos, Patentes de Auto, Empresas, y un largo etc. Es sumamente bueno cuando se quiere generar un producto en el cual la data no está lista. Súper útil, pero no lo vas a usar siempre.

- **pyyaml** (Rk: 11, ND: 142M+, Pr: 0): No vale la pena aprender más que una función para importar un yaml, esto permitirá manipular tu `yaml file` como diccionario de Python. De esta manera toda tu configuración vive en un archivo yaml, y no ensucia tus Scripts.

- **pdbpp** (Rk: 2999, ND: 173K+, Pr: 2): Es un debugger en terminal. Personalmente no me gusta el debugger de VSCode, por eso uso este. Tiene atajos de teclado y es bastante rápido. Lo recomiendo, aunque no es necesario que lo sepan utilizar.

- **holidays** (Rk: 526, ND: 4.2M+, Pr: 0): Es una librería pequeñita pero muy poderosa (se ve en su ND). Tiene todos los feriados, de todos los países de todos los años. Sólo indicas país, periodo y ya. Yo la utilice para crear features en un [Tabular Playground de Kaggle]({{ site.baseurl }}/kaggle-tps/). Muy útil en series de tiempo, pero tiene un método y ya.

- **Python-Box** (Rk: 1038, ND: 1.4M+, Pr: 0): Es súper útil, solo envuelves un diccionario con `Box()` y puedes llamar tu diccionario como `dict.key` en vez de `dict['key']`. Ahorras varios caractéres.

- **beepr** ((R), Pr: 0): Esta es una librería inútil, pero que me encantaba. Podías agregar sonidos cuando tu código fallaba o terminaba correctamente (típico sonido de Mario al pasar el nível), lo cual sacó más de una carcajada en el equipo. 

- **chime** (Rk: NA, ND: NA, Pr: 0): Sería como el equivalente en Python de Beepr. Hay otra librería más que ocupé que no recuerdo el nombre pero no funcionó tan bien.

- **Rich** (Rk: 290, ND: 9.8M+ Pr: 2): Esta es una librería muy poderosa para agregar color a tu terminal. Tiene muchas funcionalidades, barras de progreso, outputs de colores, quizás la mejor es que es posible que los errores en Python se rendericen más bonitos, para por lo menos frustrarse menor cuando algo falla. Creo que igual vale la pena invertir en una mejor experiencia de usuario cuando crees productos CLI, por lo que vale la pena aprenderla en ese caso. 

- **tqdm** (Rk: 77, ND: 38M+ Pr: 0): Tiene dos funciones interesantes, `tqdm` para envolver un For Loop y tener barra de progreso. Y otra para llamada `progress_apply` que permite barra para el apply de pandas. No te demoras nada en dominarla.

# Librerías estándar que deberías usar/conocer

- **Logging** (Rk: NA, ND: NA Pr: 1): Si vas a automatizar algo en Python, sea cual sea su uso, debes loggear todo. Logging permite generar archivos `.log` que permitirán analizar a posteriori si un Script terminó con éxito o no. Súper útil, fácil de aprender, tiene sólo un par de comandos para indicar éxito, info, warning, errores. Puedes combinar con chime y con Rich para tener un producto multicolor y sonoro. Hace más agradable la pega.

- **requests** (Rk: 5, ND: 194M+ Pr: 0): Sirve para conectarse a una API o en combinación con BeautifulSoup para obtener el HTML de un sitio WEB. Yo la he utilizado sólo para eso y no cuesta nada aprender a utilizarla, aunque quizás si debas entender el output que normalmente es un string con HTML o string con arreglos de diccionarios anidados si proviene de una API. 

- **glob** (Rk: NA, ND: NA Pr: 0): Permite revisar directorios utilizando sólo el comando `glob` y un path con expresiones regulares simples. Súper útil, por ejemplo, cuando tienes que importar muchos archivos en un sólo pandas DataFrame.

- **json** (Rk: NA, ND: NA Pr: 0): Yo lo he usado para convertir el output de requests en diccionarios y para guardar outputs como json. Sólo eso!

- **pathlib** (Rk: NA, ND: NA Pr: 2): Esta es una librería bien interesante para poder automatizar la creación de directorios y llevar tus Path de manera más sencilla. Puedes manipular Path combinandolos, creando Paths más sencillos y crear o eliminar carpetas dentro de ellos. Es fácil de utilizar y tengo ejemplos de ello en mis tutoriales de [DVC]({{ site.baseurl }}/dvc/).

- **getpass** (Rk: NA, ND: NA Pr: 0): Esta librería tiene una función llamada `getpass`, que funciona como un Text Input pero con los caractéres ocultos. Útil para ingresar data que no quieres que se vea, pero no la encripta ojo.

## Uff
{: .no_toc }

Y con esto terminamos. Debo decir que este es al artículo que más trabajo me ha dado. Y demoré cerca de dos meses en escribirlo. Voy a tratar de ir llenando esto con el tiempo a medida que vaya probando más cosas. Algunas de las tecnologías que se me quedaron en el tintero porque no cumplen los requisitos exigidos arriba:

* Varios servicios AWS
  * Sagemaker
  * AWS Lambda
  * API Gateway
  * Step Functions
* [PRegex](https://pregex.readthedocs.io/en/latest/)
* [Dagshub](https://dagshub.com/)
* [poetry](https://python-poetry.org/). Por alguna razón me da terror instalarla, aunque he leído bastante de ella.
* [fancyimpute](https://pypi.org/project/fancyimpute/)
* [NGBoost](https://stanfordmlgroup.github.io/projects/ngboost/)
* [SKLego](https://scikit-lego.readthedocs.io/en/latest/)
* [tabnet](https://github.com/dreamquark-ai/tabnet)
* [UMAP](https://umap-learn.readthedocs.io/en/latest/)
* [Segmentation Models](https://github.com/qubvel/segmentation_models.pytorch)
* [NannyML](https://www.nannyml.com/)
* [DGL](https://www.dgl.ai/)
* [BioPython](https://biopython.org/)
* Ecosistema Pytorch :
  * [torchtext](https://pytorch.org/text/stable/index.html)
  * [torchaudio](https://pytorch.org/audio/stable/index.html)
  * [torchgeo](https://torchgeo.readthedocs.io/en/stable/)
  * [Pytorch Geometric Temporal](https://github.com/benedekrozemberczki/pytorch_geometric_temporal)
  * [torchrec](https://pytorch.org/torchrec/)
  * [Pytorch Video](https://pytorchvideo.org/)

Espero que esto sea de utilidad para tenerlo como referencia a la hora de enfrentar distintos problemas en Ciencia de Datos.

Nos vemos,

[**Alfonso**]({{ site.baseurl }}/contact/)
