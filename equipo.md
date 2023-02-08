---
permalink: /equipo/
layout: page
title: "Mi Equipo"
subheadline: ""
teaser: ""
---
<!-- ...and learn more at the same time. You can message me at the [contact page]({{ site.baseurl }}/contact/). -->

{% comment %} Including the two pictures is a hack, because there's no way use the Foundation Grid (needs a container with row) {% endcomment %}
![picture of me]({{ site.urlimg }}widget2-equipo_2.jpeg){: .left .show-for-large-up .hide-for-print width="400"}
![picture of me]({{ site.urlimg }}widget2-equipo_2.jpeg){: .center .hide-for-large-up width="500"}

<q>Actualizado al 2023</q>

Para hacer ciencia de Datos hoy en día es necesario tener un equipo que te acompañe y que tenga algunos requerimientos mínimos. Obviamente la mayoría de las empresas te van a entregar un computador y contra eso no hay mucho que hacer. Ahora, en mi opinión la mayoría de los equipos entregados no cumplen con lo que uno necesataría. Muchas empresas deciden utilizar computadores baratos y disponibilizar algún servicio cloud en el caso de necesitar más poder computacional y eso está perfecto. Ahora independiente de si tienes un equipo personal (como es mi caso) o si tienes uno en la nube, lo ideal es poder contar con todo lo que necesitas. En mi caso prefiero un computador personal, lo utilizo no sólo en mis trabajos sino también en mis proyectos personales. Además no hay nada como tener tu propio computador para hacer todas las pruebas que quieras, dejar corriendo en la noche sabiendo que no tendrás que pagar un dineral en maquinas virtuales. Ahora, los tutoriales muestran que todo es <mark>color de rosa</mark> (si es que trabajas con Windows, las personas que usan Mac saben el dolor de cabeza que es dejar una configuración estable, que mal compu es un Macbook), pero la verdad no es tan así. Y además, considero que personas que se quieran tomar esto en serio no deberían estar utilizando un SO como Windows donde muchos recursos se desperdician y no hay un buen terminal para trabajar y reproducir tranquilamente. 

Por lo tanto, decidí hacer este review/tutorial, mostrando las características de mi equipo, pero también todos los procesos para instalar los sistemas operativos que actualmente uso y obviamente todos los errores no esperados que aparecen a lo largo de la instalación y el setup de todo. Realmente me costó mucho dejar todo funcionando, pero desde ahí en adelante mi computador jamás ha dado problemas, incluso cuando lo exijo al máximo.

Por completitud voy a dejar también mi antigua configuración explicando claramente que es **legacy**, ya que actualmente estoy utilizando PopOS, una variante de Ubuntu que me tiene muy contento, pero al menos por 3 años Ubuntu me acompaño sin ningún problema.

{% include alert info='Considero que es especialmente imporante contar con equipos que den el ancho, en especial porque las empresas piensan que entregando el Mac más caro es suficiente, o un supercomputador con Windows, pero la verdad es que cada vez me convenzo más que Linux es la mejor opción. No sólo porque es gratis sino porque también la mayoría de los servidores corren en algún sabor de Linux, por lo que un Data Scientist serio debería poder manejarse con la terminal y no tener miedo a cambiar de sistema operativo, mal que mal somos una rama de la informática, hay que saber arreglar computadores XD' %}

Entonces, ¿Cuáles son los aspectos más importantes al momento de comprar un computador? (o en su defecto, al momento de pedir un computador a tu empleador) La verdad es que esta es una pregunta bien dificil de responder, pero en general, se puede resumir en alto poder de cómputo. ¿Qué es lo que en particular yo estaba buscando? En mi caso, quería un computador con características ***gamer***, no porque juegue, sino porque están mejor preparados para una alta demanda de recursos, en partícular, la ventilación. Dado que ocupar muchos recursos generalmente genera un aumento en la temperatura es que este aspecto es bastante importante. Además me interesaba mucho la portabilidad, no busco un computador liviano, pero sí poder moverme, y es por eso que decidí que un Laptop era una buena ~~la mejor~~ opción. 

{% include alert warning='Mis computadores están pensados para alguien que disfruta compitiendo en Ciencia de Datos y que paso entrenando modelos muy seguido. Por lo tanto, está muy enfocado a tener mucho poder en términos de cómputo y memoria. Es posible que no sea necesario tanto, pero por lo menos con esta configuración siempre he dado el ancho para entrenar todo tipo de modelos, desde Visión Computacional, Grafos y NLP. Pronto tutoriales acerca de eso...!' %}

{% include toc.md %}

## Laptop

Decidí ir por un Laptop Lenovo, la verdad es que si bien he leído algunos comentarios de algunos problemas en torno a su temperatura, luego de dos años no he tenido ningún problema. Debo decir que se me hizo muy complicado ver qué Laptop comprar, primero, porque no hay tanto para elegir en Chile, y segundo, porque muchos de los reviews en Youtube muchas veces son muy enfocados en gaming y terminan siendo muy críticos, por lo que siempre queda la inseguridad si será un buen computador. Finalmente hay que arriesgarse, y en mi caso creo haber tomado una buena decisión.

Ahora, cuáles son las características de mi laptop?:

![picture of me]({{ site.urlimg }}equipo/neo_v2.png){: .center}

* **Legion-7i de 15.6"**: Principalmente me gustan los notebooks grandes y ojalá con teclado numérico. No es tan portable, en el sentido que tiene un cargador gigante y debe pesar unos 2.5 kilos en total, pero a mí eso no me molesta. Solía tenerlo con Ubuntu, pero desde el 2022 me cambién a PopOS. Una de las grandes ventajas que ofrece PopOS es que tiene control de energía. Con Ubuntu no lograba tenerlo más de 1 hora sin cargador. Con los Power Settings en balanceado, he durado cerca de 3 horas, probablemente se pueda más en Modo Ahorro de Energía.

* **32GB de RAM @ 2666MHz**: Partí con 16GB y en la mitad de la competencia Binnario en el 2020 me quedé corto y tuve que comprar más, 32GB es lo máximo que soporta el Legion 7i y anda bien, en caso de requerir más está **Jarvis**. 

* **Procesador Intel i7-10750H @5.00GHz**: Aquí lo principal que hay que fijarse es en la letra final del procesador. H, es por High Performance, por lo tanto casi no sufre throttling. En el computador de mi pega en Cencosud, y de hecho en casi todas las que me han pasado Laptops, tenía un procesador terminado en U, que son la serie de ahorro de energía. El problema de estos procesadores es que siempre hacen throttling, es decir, se frenan cuando alcanzan mucha temperatura, esto para ahorrar más energía lo cual no es algo deseado cuando se quiere utilizar el compu a máxima capacidad.  

* **GPU Nvidia RTX 2070 Max-Q**: Esto fue un capricho, quería una Tarjeta que tuviera tensor cores para ver si se sentía la diferencia, y la verdad es que sí se siente.  Lo bueno de tener GPU en el Laptop es que puedo hacer pruebas pequeñas bien rápido y en caso de requerir más poder me cambio al PC. 

* **Disco Duro NVme 512GB**: Acá me conformé con lo que había, 512GB para mí es más que suficiente, pero me preocupé que el disco duro fuera NVme para tener mejor performance lectura-escritura. Hoy en día el almacenamiento no es problema, con discos gigantes en Dropbox o Google Drive, incluso S3 si es que tienes AWS es más que suficiente. Igualmente tengo un disco externo de 1TB para guardar cosas más pesadas y a las cuales no necesito acceder rápidamente. 

Creo que esa son las características principales que uno debiera mirar al momento de elegir un computador, en términos de poder computacional anda bastante bien, lo que siempre me ha tenido preocupado es el tema de la ventilación. Ahora otras cosillas que son menos importante son las luces (todo buen computador gamer tiene que tener luces RGB) y su monitor que tiene tasa de refresco de 240Hz que permite muy buena definición y fluidez. 

{% include alert alert='Para sacarle el mayor provecho al laptop siempre debe estar en una temperatura adecuada para evitar el throttling. El throttling es un mecanismo de autocuidado que puede tener tanto el procesador como la GPU para bajar su rendimiento con el fin de disminuir su temperatura. En mi caso la temperatura de mi GPU que es lo que más he recargado no ha pasado de los <mark>60°C</mark>. Es una temperatura alta, pero al menos la refrigeración hace su trabajo y nunca he sentido que me voy a quemar o algo por el estilo. Al menos la RTX 2070 tiene una temperatura máxima antes de throttling de <mark>93°C</mark>, por lo que aún tengo bastante margen.' %}

Repito, <mark>no he tenido problemas de temperatura</mark> pero siempre es algo que me asusta en especial cuando quiero dejarlo a máxima potencia. La ventilación que tiene es altamente criticada en foros, la cual es un vapor chamber, pero digan lo que digan, cumple.

![picture of me]({{ site.urlimg }}equipo/vapor.png){: .left .show-for-large-up .hide-for-print width="500"}
![picture of me]({{ site.urlimg }}equipo/vapor.png){: .center .hide-for-large-up width="500"}

Hay una característica que en particular me gustó, que son los distintos modos que permiten un mayor nivel de ventilación. Modo Performance, cambia el color del botón de encendido a rojo y genera ventiladores a toda potencia, a mí no me molesta, porque prefiero que haga la pega de enfríar el PC (y aparte uso audífonos).
Color blanco, es media potencia, ni fu ni fa, y color azul, baja potencia, probablente útil para ahorrar energía pero que yo sólo utilizo cuando no tengo mi cargador a mano.

Otro aspecto que me gustó mucho es que tiene bastantes conectores: 4 USB, 2 USB-C (1 con thunderbolt), conexión HDMI, y conexión a carga rápida, en realidad carga tan rápido como se descarga. Esto me permite otro de mis caprichos que es tener un setup con 4 monitores: 1 por HDMI, 1 por USC-C, 1 por Thunderbolt y el integrado del Laptop.

{% include alert todo='En mi caso el 95% del tiempo está enchufado y en modo performance 😀, lo cual hace que suenen más los ventiladores. A algunos les molesta, pero ayuda a cuidar el computador.'%}

> Probablemente mi caso no es el más recomendable, yo decidí ir por un laptop gamer, no es la mejor opción, pero primeramente estaba pensando en portabilidad. Debido a que ya tomé una decisión en dedicarme de manera full al Machine/Deep Learning es que ~pretendo invertir en un PC de escritorio con más poder~ invertí en un PC de escritorio con mucho más poder, pero como explicaré luego lo utilizo más en modo servidor y al cual bauticé como JARVIS.

##  Y también armé un PC 

Bueno, luego de ganar el Itaú Binnario pensé que era bueno entrar en algún proyecto de armado de computadores. Como cuento [**acá**]({{ site.baseurl }}/concurso/), pasé por muchos problemas durante la competencia debido a la falta de recursos computacionales.  Si bien cree un artículo para hablar del armado del compu, nunca tuvo tanta visibilidad así que decidí agregarlo acá. Jarvis, es un computador armado desde cero y el cuál utilizo como servidor remoto conectándome por SSH. Las características de Jarvis son las siguientes:

### GPU: El corazón del PC
![picture of me]({{ site.urlimg }}projects/jarvis/gpu-removebg-preview.png){: .left .show-for-large-up .hide-for-print width="500"}
![picture of me]({{ site.urlimg }}projects/jarvis/gpu-removebg-preview.png){: .center .hide-for-large-up width="250"}

La GPU no es para nada la pieza más importante al momento de armar un PC, pero en este caso era fundamental porque es un PC para hacer modelos de Machine/Deep Learning. Hace un tiempo he estado siguiendo varios youtubers Kaggle GrandMasters y todos ellos hablaban de la importancia de una buena tarjeta gráfica. 
Al momento de comprar mi Legion-7i, pude notar la diferencia de entrenar en una RTX 2070. Realmente es mucha la diferencia. Pero al empezar a trabajar con modelos más grandes el gran problema que puede tener una GPU es con cuantos datos puede trabajar de manera simultánea.

Aquí es cuando la RTX 3090 entra en juego. Es carísima, es lo más caro que he comprado y más cara que mi Laptop, pero vale completamente la pena. Yo compre la <mark>ASUS ROG STRIX RTX 3090</mark>, hasta ahora la que mejor desempeño ha entregado.

![picture of me]({{ site.urlimg }}projects/jarvis/gpu2-removebg-preview.png){: .right .show-for-large-up .hide-for-print width="500"}
![picture of me]({{ site.urlimg }}projects/jarvis/gpu2-removebg-preview.png){: .center .hide-for-large-up width="250"}

La RTX 3090 tiene:

* 10496 Cuda Cores.
* 936 GBps de Bandwidth.
* 328 Tensor Cores (3era Gen) (muy útil para Deep Learning) y 82 Ray Tracing Cores (útil para Renderizado).
* Posibilidad de NVLink para MultiGPU. Algo que me gustaría explorar cuando tenga dinero para otra tarjeta. 
* 24GB GDDR6X. Probablemente la gran ventaja de la tarjeta, 3 veces más memoria que la RTX 2070.

La gran diferencia de esta tarjeta es precisamente, la cantidad de memoria RAM disponible para cargar datos. Su gran impacto será entonces en modelos de Deep Learning, por lo que independiente de todos los procesos en paralelos que se pueden generar, y de las operaciones en tensores, permite crear Batches de entrenamiento mucho más grandes dando rápidez y estabilidad en el cálculo de gradientes. 

Este fue el primer componente adquirido, justo antes del encierro por pandemia tuve la oportunidad de ser de los primeros en adquirir la RTX 3090 (esta tarjeta ya fue superada en parte por la RTX 4090 pero que sigue siendo tremendamente potente por su relación precio/calidad).

### CPU
![picture of me]({{ site.urlimg }}projects/jarvis/ryzen-removebg-preview.png){: .left .show-for-large-up .hide-for-print width="400"}
![picture of me]({{ site.urlimg }}projects/jarvis/ryzen-removebg-preview.png){: .center .hide-for-large-up width="250"}

En este caso, me la jugué por lo que está ganando popularidad y me fui por Ryzen. Le instalé un <mark>Ryzen 7 5800X</mark> de 3era generación. Es un procesador que ha tenido muy buenos reviews, porque es extremadamente potente. Tiene 8 cores y 16 threads con velocidad base de 3.8Ghz y he visto que le han sacado hasta 5GHz multicore (eso sí congelándolo), pero si no llega a los 4.7GHz.

Principalmente esto beneficia en el número de procesos paralelos y tiene directo impacto con modelos de Machine Learning de Scikit-Learn. Debido a que éstos corren en CPU, más threads permiten paralelizar Cross Validation, OnevsRest o algunos ensambles. El único punto a considerar es que es un procesador grande que se calienta al parecer bastante, por lo que es necesario invertir en buena refrigeración.


### Motherboard

![picture of me]({{ site.urlimg }}projects/jarvis/mobo-removebg-preview.png){: .left .show-for-large-up .hide-for-print width="500"}
![picture of me]({{ site.urlimg }}projects/jarvis/mobo-removebg-preview.png){: .center .hide-for-large-up width="250"}

Acá también me la jugué por algo gama media, si bien podría haber adquirido algo superior, esta placa ha tenido excelentes reviews y está catalogada como la mejor en calidad/precio: <mark>TUF X570-PLUS Wifi</mark>. Estoy muy contento con esta placa, pero quizás me faltó un poco de research acá. Es tremenda placa y no me ha dado ningún problema sólo que hay cosas que no me fijé:

* Tiene Wifi5, el cuál es muy bueno y no me ha dado problemas de conectividad, pero existe Wifi6,
* Tiene capacidad de 2 discos duros M.2, pero sólo una conexión con disipador. A mí no me molesta, tengo sólo un disco, pero si quisiera agregar otro quedará sin disipación.
* Tiene sólo una entrada PCIe reforzada, lo cual es más que suficiente, pero si quisiera otra GPU probablemente tendré problemas.
* No caben bajo ninguna circunstancia 2 RTX 3090, pero está bien hoy en día existen conectores para sobrellevar el problema de espacio. 
* No es compatible con Ryzen 3era Generación directamente, hay que actualizar BIOS.

Pareciera una tarjeta terrible, pero no, es tremenda tarjeta.

{% include alert warning='Si van a comprar una tarjeta madre para Ryzen de 3era generación hay que fijarse que sean 3rd-Gen Ready. En ese sentido muchas gracias a la gente de [Nice One](https://n1g.cl/Home/) quienes tienen la tarjeta a muy buen precio y me actualizaron la BIOS. No tuve ningún problema para ensemblar.'%}

Lo principal y que es algo que estoy en proceso de implementar es que es compatible con Wake-on-LAN, es decir me permite encender el PC a través de Internet. Muchos puntos por eso!! Aunque debo decir que después de casi dos años todavía no logro hacerlo XD.


## Refrigeración

![picture of me]({{ site.urlimg }}projects/jarvis/darkpro-removebg-preview.png){: .right .show-for-large-up .hide-for-print width="500"}
![picture of me]({{ site.urlimg }}projects/jarvis/darkpro-removebg-preview.png){: .center .hide-for-large-up width="250"}

Este apartado es el tema que menos domino, y del que menos entendía su importancia. En mi Laptop empecé a notar que siempre está muy caliente. Esto porque tiene un Intel i7 y la RTX2070. En estado base la temperatura de tarjeta de video es de 46°-48°. Por lo tanto investigando, encontré un tremendo post en el que se aconsejaba ventilación por Aire. Quizás en el mundo gamer la ventilación liquida la lleva y es mucho más llamativa, pero este tipo es Phd, haciendo modelos de Deep Learning todo el día y decía que para una GPU mejor aire, así que aire no más.

Bueno, porque estaba en oferta y porque leí buenos comentatios me fui por el <mark>be Quiet Dark Pro Rock 4</mark>. Algunos foros decían que el Noctua es mejor, pero por poquito y la verdad que conseguir un Noctua se me hizo imposible por la pandemia. Pero estoy contento, no sólo porque mi PC está extremadamente silencioso (una de las ventajas de este disipador) y porque la CPU en estado base está en 46°, que por lo que leí está bastante bien.

![picture of me]({{ site.urlimg }}projects/jarvis/fan.png){: .right .show-for-large-up .hide-for-print width="500"}
![picture of me]({{ site.urlimg }}projects/jarvis/fan.png){: .center .hide-for-large-up width="250"}

Adicionalmente tengo 4 ventiladores extra, uno que venía con el case que ayuda al Dark Pro a sacar aire, y 3 que inyectan aire frío. Aquí me fui por los <mark>Corsair AF140</mark>. Los compré porque era de los más barato que pillé, y me topé con que son top de línea y se nota, al comenzar a funcionar pude sentir la cantidad de aire que inyectan. En estado base la RTX 3090 está a 28°. Más que contento.

Un único inconveniente que no me fijé, es que los conectores para ventilador de la placa madre eran de 4 pines y estos ventiladores son de 3 pines. Afortunadamente, el case traía un convertidor (creo que era del case, no lo sé la verdad) y todo funciona muy bien, pero esto es algo que no me fijé y que pudo impedir que el PC estuviera funcionando en su mejor forma.


## Disco Duro (SSD)

![picture of me]({{ site.urlimg }}projects/jarvis/ssd.png){: .left .show-for-large-up .hide-for-print width="400"}
![picture of me]({{ site.urlimg }}projects/jarvis/ssd.png){: .center .hide-for-large-up width="250"}
Me fui por un Samsung 970 Evo Plus. Es lo mejor que pude encontrar, tiene velocidad de escritura y lectura bien altas (3500 MB/seg y 3300 MB/seg respectivamente) y para mí es más que suficiente. El disco duro no suele ser un componente que tenga gran impacto excepto para cargar datos al momento de entrenar, en especial en los DataLoaders en Pytorch. 

<br>
<br>
<br>

### RAM

![picture of me]({{ site.urlimg }}projects/jarvis/ram.png){: .right .show-for-large-up .hide-for-print width="400"}
![picture of me]({{ site.urlimg }}projects/jarvis/ram.png){: .center .hide-for-large-up width="250"}

Acá también se recomendaba algo no muy caro, ya que la velocidad de RAM no es de gran impacto al momento de entrenar modelos. En general, dado que los modelos de Machine Learning se cargan en memoria al momento de paralelizar era importante tener al menos 32GB y las que elegí fueron las <mark>T-Force Delta</mark>. Son DD4 a 3200MHz y aparte tienen luces, que no me mataba, pero se ve bien bonito, sino miren la foto del inicio.

Luego de un par de competencias en Kaggle me dí cuenta que 32 GB no era suficiente y mi querido hermano me regaló más RAM, unas Kingston Fury con la cual llegué a los 64 GB a 3200MHz. Por ahora, cero problemas de memoria. La verdad no creo que llegue a utilizarla toda, pero en general necesito del orden de 40 GB.

## Fuente de Poder

![picture of me]({{ site.urlimg }}projects/jarvis/psu-removebg-preview.png){: .right .show-for-large-up .hide-for-print width="300"}
![picture of me]({{ site.urlimg }}projects/jarvis/psu-removebg-preview.png){: .center .hide-for-large-up width="250"}
Esto era también super importante. Si bien no es algo que uno usa directamente, los requerimientos de energía de este sistema son bastante altos. Para una RTX 3090 NVidia recomienda una fuente de al menos 750W. Varios reviews dicen que uno anda bien justo, y que se han presentado cortes al exigir a su tope a la Fuente. Además tiene que ser de calidad porque no quiero quemar todos mis componentes por abaratar costos acá. Por eso me fui por la <mark>EVGA GQ 1000</mark>, de 1000W. Si es que quisiera agregar más cosas puedo hacerlo sin miedo. Además tiene certifición GOLD y no puede venir con más cables. Una de las mejores sugerencias del setup, y en caso que quisiera eventualmente agregar otra GPU creo que aguantaría bien.

<br>
<br>
<br>

### Case

![picture of me]({{ site.urlimg }}projects/jarvis/case-removebg-preview.png){: .left .show-for-large-up .hide-for-print width="400"}
![picture of me]({{ site.urlimg }}projects/jarvis/case-removebg-preview.png){: .center .hide-for-large-up width="250"}
Todas estos componentes no caben en cualquier case. Por lo tanto, me recomendaron utilizar un Full Tower, y me fijé que fuera muy grande, que cupiera todo sin problemas y que permitiera la entrada de mucho aire. Por eso escogí el <mark>Cougar Panzer Max-G</mark>. Dentro de las características que me gustaron es que trae todo lo que necesitas (muchos tornillos, piezas de ensamblaje y sobre todo el adaptador para los ventiladores que me salvó la vida), tiene vidrio templado para ver los componentes con RGB, permite controlar la intensidad de los ventiladores, permite como hasta 8 ventiladores (si tengo problema de temperatura tiene mucho espacio para seguir refrigerando) tiene 4 USBs adicionales, conector de micrófono, audífonos y un gancho para colgarlos.
Además permite un ensamblado super ordenado, trae compartimento aislado para la Fuente de Poder y separación para la gestión de cables, además se siente muy muy firme.


<br>
<br>
<br>
<br>
<br>
<br>
<br>


## Teclado

Para el caso de mi teclado quise hacer un apartado adicional, ya que es bastante especial. Mi teclado actual es el Corne Keyboard y es el segundo teclado mecánico que tengo. Si bien estaba muy contento con mi primer teclado, me empezó a llamar mucho la atención el uso de teclados divididos. Luego de ver varios videos de [Hola Mundo](https://www.youtube.com/c/HolaMundoDev?app=desktop) comencé a adentrarme en el mundo del Touch Typing y la verdad que los teclados stagger comenzaron a darme mucho dolor de muñecas, debido a la curvatura necesaria para teclear con todos los dedos. Algo que este teclado soluciona. Tengo este teclado hace poco más de un mes. Ya recuperé mi velocidad de escritura pero me cuesta bastante programar aún. Llegar a una configuración cómoda me tomó cerca de 3 semanas y aún me cuestan los símbolos. Pero es algo que sufrí con mi anterior teclado por lo que tengo esperanza. Es un teclado de 42 teclas, con switches Gateron Brown, que a diferencia de los Cherry Red que tenía antes han ayudado mucho a disminuir la cantidad de errores que cometo al tipear a alta velocidad. 

![picture of me]({{ site.urlimg }}projects/jarvis/corne.png){: .right .show-for-large-up .hide-for-print width="500"}
![picture of me]({{ site.urlimg }}projects/jarvis/corne.png){: .center .hide-for-large-up width="250"}

El teclado es espectacular ya que fue hecho totalmente a medida por los chicos de [Zone Keyboards](https://zonekeyboards.cl/) con mi logo personalizado, teclas especiales, y un case de madera de lenga muy elegante. Creo que el único pero que tiene es que como utilizo touch typing, no tiene los bumps en las teclas f y j y a veces me cuesta llegar al home row (pero de a poco ya me he ido acostumbrando ya que las referencias en este caso se dan a través de los pulgares). Pero aparte de eso las configuraciones posibles son espectaculares, casi no muevo las manos de mi posición base y tengo muchos atajos y acordes (son como shortcuts pero que se pueden ir escribiendo tecla a tecla en vez de todos al mismo tiempo, lo cual lo hace bastante más cómodo) personalizados. Otra de las características que me encanta es que pude configurar mi teclado como si fuera VIM pero sin el molesto modo insert con la letra *i* y estando como capa base las letras en vez del desplazamiento, y utilizando VSCode (estas eran las principales razones por la que VIM no me terminaba de convencer). 

Tiene 6 botones para pulgares lo cual permite agilizar los típicos shortcuts ya que no tengo que sacar nunca mis manos del homerow, ni siquiera para un `Ctrl+C` o `Ctrl+V`. 

### Teclado Legacy

Este es un teclado mecánico que compré por Kickstarter que de verdad anduvo muy bien durante los 2 años que lo usé. Es el Epomaker GK68XS y su gracia es que tiene algunas características especiales:

![picture of me]({{ site.urlimg }}equipo/teclado.jpg){: .center}

Primero es un teclado 65% por lo que es más compacto (aunque no tanto como el Corne que es 40%). Tiene luces, lo cual ayuda en la oscuridad, pero no las uso mucho. Puede conectarse por USB tipo C o por Bluetooth lo cual es bastante cómodo, tiene Cherry Red switches lo cual lo hace muy agradable al tipeo, pero lo que más me gusta es que es 100% configurable. Como ven tiene 3 barras espaciadoras lo cual me permite teclas extras que yo puedo elegir, además de atajos multimedia, 3 capas de teclas y creación de macros, lo cual es muy útil para poder programar.

Todas las capacidades de este teclado y más son posibles con mi Corne. El gran contra del teclado es que tiene las filas staggered y al usar touch typing me daba mucho dolor de muñecas en mi mano izquierda que es la que más sufre. 

## El Mouse

También hice algunos cambios acá, utilizo un trackball en el cual no desplazo el mouse sino una ruedita que está encima. Si bien me encantaría cambiar este mouse por uno inalámbrico, he gastado mucho dinero en otras cosillas que en verdad han hecho mi vida más simple.


![picture of me]({{ site.urlimg }}equipo/trackball.png){: .center width="250"}

### Mouse Legacy

Solía utilizar un mouse vertical marca Zelotes, me encanta, muy útil, y tiene algunas teclas extras que me permiten navegar más rápido. Pero bajar el mouse forzaba en mi muñeca un movimiento muy antinatural que terminó derivando en tendinitis, por eso el trackball. Pero debo decir que era más mi forma de mover el mouse que el mouse en sí, por lo que si quieren un buen mouse vertical (y además barato) el Zelotes es buenísimo.

## Monitores

Este también es un apartado en el que decidí invertir y el cual se vió muy beneficiado por la gran cantidad de conectores del Laptop. Acá tengo dos monitores de 27" Samsung y un tercer monitor Samsung de 21.5" el cual uso de manera vertical. Por qué 3 monitores? Porque puedo. Estoy recién comenzando a ver los beneficios ya que en la práctica tengo 4 monitores: el Laptop y 3 más. Tengo dos principales y dos panorámicos ocasionales. Esto no está exento de problemas. El rango visual es demasiado amplio y he estado sufriendo dolores de cuello, pero no sé hasta que punto esto se debe sólo a los monitores y no a mi estrés post-universidad. 

{% include alert success='Luego de casi 2 meses usando esta configuración puedo decir que realmente me encanta, y mis dolores de cuello se han ido. Nuevamente, porque eran más por la alta carga del año pasado. Si bien 4 monitores es algo exagerado, vale completamente la pena tener más de un monitor.' %}

## Escritorio

Quizás esto es lo más caro en lo que he invertido, pero dado los dolores de espalda y cuello por probablemente la gran cantidad de horas que paso en frente del computador decidí ir por un <q>Standing Desk</q>. Me encanta, me ayuda a no estar todo el día sentado, aportando a diminuir el sedentarismo propio de mi oficio. Pero además creo que ayuda bastante para hacer clases de manera más dinámica. Y no sé si es placebo pero ha ayudado bastante a aliviar los dolores de espalda principalmente. 

Mi setup actualmente se ve algo así:

![picture of me]({{ site.urlimg }}equipo/equipo_final.jpeg){: .center .show-for-large-up .hide-for-print width="600"}
![picture of me]({{ site.urlimg }}equipo/equipo_final.jpeg){: .center .hide-for-large-up width="250"}


## Otros

También tengo unos audífonos bluetooth marca Soundcore que andan muy bien, un poquito de carga y duran muchas pero muchas horas. Hace bastante que el Bluetooth dejó de funcionar pero tengo otros exactamente iguales nuevitos de paquete que pronto abriré. 

Para armar el setup tengo 3 brazos hidráulicos que me permiten levantar tanto los monitores como el laptop a mi voluntad, si me da por trabajar de pie podría hacerlo (aunque ahora esa tarea la hace el escritorio). Mi silla es Nitro Concepts S300. Me salió muy cara, me costó mucho que llegara, no es el color que más me gusta (obvio había falta de stock en la pandemia) pero desde que la compré casi no tengo dolor de espalda (muy rara vez me molesta pero porque no me estoy apoyando correctamente), y en verdad paso 10 hrs diarias en el PC, a veces más. Las características más importantes son:

* Pistón clase 4, y armazón de acero creo que soporta hasta 135 kgs.
* Reclinable full y muy blandita, no sirve de mucho, pero es rico hasta dormir acá.
* Apoya brazos 3D, podría ser 4D, lo extrañé, porque está pensada para gente muy grande, y yo no soy tanto.
* No es de cuerina, es de tela, por lo que no transpiro nada, pero... se le pega el pelo de gato, pero no se puede pedir todo. 
* Pero por sobre todo, calidad alemana, está muy bien termindada, como se une la tela al asiento, el tipo de tela, las costuras. Mis gatos la han atacado y ha resistido muy bien.


Otros accesorios son una cubierta tipo mousepad gigante comprado en Ikea, el [Rissla](https://www.ikea.com/cl/es/p/rissla-protector-de-escritorio-negro-40246156/). Como pueden ver es muy grande, de base metálica y con cobertura de cuero. Es muy elegante y le da ese toque estético para que no parezca un setup puramente gamer. Y para la paz, una plantita, falsa, también de Ikea pero que la da un toque hogareño. Creo que eso es todo lo que tengo por ahora y que realmente me pone bastante feliz cuando tengo que sentarme a trabajar. También invertí en una camara web Logitech, tengo planeado eventualmente lanzar un canal de Youtube, si es que logra suficiente tracción por lo que de a poco debo ir invirtiendo en iluminación (tengo un aro de luz que anda bastante bien) y también un microfono para streaming el cual compré cuando me puse a grabar cursos para desafío Latam pero que no se ven la imagen. Afortunadamente todo el setup como logró una combinación rojo con negro por lo que se ve bastante coherente. 

> Ya, de vuelta al compu.


## Sistema Operativo

He sido una persona que he utilizado Windows la mayor parte de la vida, y la verdad es que nunca me había quejado hasta que comencé a hacer modelos más grandes, y me dí cuenta el desperdicio de recursos de Windows. Además que al consumir toda la RAM el computador se cuelga y no vuelve más, eso si es que no recibes alguna pantallita azul.

Es por eso que decidí jugármela e instalé Ubuntu, en mi caso 20.04 LTS. Pero desde el 2022 me cambié a PopOS. 


### Documentación para mi yo del futuro, pero Legacy

{% include alert tip='Acá hay un abanico de posibilidades y sabores diferentes de Linux, decidí cambiarme por algunas recomendaciones que leí en la web y porque en verdad todos decían que me haría la vida más fácil en términos de compatibilidad. Pero NO, muchos problemas ocurrieron.'%}

Luego de seguir las recomendaciones de [David Adrian Quiñones](https://davidadrian.cc/), me decidí por Ubuntu y varios otras de las sugerencias que hace (No todas).

{% include alert tip='En mi caso hice dual boot, como se explica en el video, esto implica dejar Windows instalado en caso de necesitarlo. Creo que nunca he tenido que usarlo, pero siempre puede salvar de un apuro.'%}

<div class='embed-youtube'>
{% include youtubePlayer.html id="-iSAyiicyQY" %}
</div>

{% include alert alert='Acá comienzan los problemas, si bien, el video muestra que instalar Ubuntu es una maravilla y no hay ningún problema asociado, la verdad es que no es así. Acá les muestro varios inconvenientes que tuve al momento de instalar Ubuntu.'%}

### No se reconocen los drivers de Video

{% include alert info='Yo ya había recién comprado un Lenovo Legion 5 que tuve que devolver porque tuve problemas con los drivers, y además un slot de RAM estaba dañado por lo que en todo momento pensé lo peor.'%}

Al comenzar con el Booteable, lo primero que veo luego del Menú de instalar Ubuntu es lo siguiente:

![picture of me]({{ site.urlimg }}equipo/nvidia_fail.jpeg){: .right .show-for-large-up .hide-for-print width="500"}
![picture of me]({{ site.urlimg }}equipo/nvidia_fail.jpeg){: .center .hide-for-large-up width="500"}

Obviamente uno entra en pánico al ver sólo manchas en la pantalla, y lo primero que hice fue googlear acerca de este problema. Ahí encontré que esto era algo <q>normal</q> cuando se tiene una GPU Nvidia y era sencillo de solucionar, pero era mi primera experiencia con Ubuntu y con trabajo en Terminal, por lo que siempre da como cosa modificar elementos por línea de comando.

La solución según los foros era desactivar los drivers por defecto, que normalmente son drivers open source que se llaman algo así como `Nouveau` y que no funcionan.

Para hacer eso hay que hacer lo siguiente: Justo en la pantallita que pide instalar Ubuntu hay que presionar la tecla `E`. Esto nos llevará a una pantalla que dice lo siguiente:

<br>

```shell
set params 'Ubuntu'
      set gfxpayload=keep
      linux    /casper/vmlinuz file=/cdrom/preseed/ubuntu.seed maybe-ubiquity quiet splash ---
      initrd   /casper/initrd
```

{% include alert success='Para solucionarlo sólo hay que agregar la palabra nomodeset entre `splash` y `---` luego hay que presionar `F10` o `Ctrl+X` para reiniciar. Ahora sí hay que darle a instalar.'%}

```shell
set params 'Ubuntu'
      set gfxpayload=keep
      linux    /casper/vmlinuz file=/cdrom/preseed/ubuntu.seed maybe-ubiquity quiet splash nomodeset ---
      initrd   /casper/initrd
```
{: title="Solución para desactivar Drivers"}

{% include alert warning='Esta desactivación es sólo temporal y para evitar este problema al iniciar es que una vez que Ubuntu está correctamente instalado se necesita instalar los propietary drivers de Nvidia, lo cual Ubuntu hace de manera casi automática.'%}

Van a seguir apareciendo caractéres raros pero si ves algo así, se puede respirar tranquilo:

![picture of me]({{ site.urlimg }}equipo/booting.jpeg){: .center}

Una vez pasada esta primera etapa hay que seguir con el proceso normal, y no <q>debieran haber más errores</q> pero para mi mala suerte apareció otro obstáculo:

### RST (error no tan frecuente)

{% include alert alert='La verdad no tengo muy claro que es la tecnología `Intel RST`, pero al parecer varios de los últimos computadores gamers vienen con esta tecnología incluida. Según mi investigación esta tecnología permite una sincronización cuando hay  más de un disco duro, que no es mi caso, por lo que la verdad, no me beneficiaba en nada, es más, me impedía instalar Ubuntu.'%}


![picture of me]({{ site.urlimg }}equipo/rst.jpeg){: .right .show-for-large-up .hide-for-print width="500"}
![picture of me]({{ site.urlimg }}equipo/rst.jpeg){: .center .hide-for-large-up width="500"}

Para desactivar RST la verdad es que es bastante sencillo, y hay dos formas de hacerlo, una más UI y otra por línea de comando, yo elegí la línea de comando porque como que en cierto sentido ya estaba perdiendole el miedo. El tema es que esta desactivación hay que hacerla en Windows, por lo que fue super bueno hacer el Dual Boot en vez de eliminar Windows de frentón.

Los detalles completos los pueden encontrar [aquí](https://askubuntu.com/questions/1233623/workaround-to-install-ubuntu-20-04-with-intel-rst-systems), yo seguí el `choice 2`, por lo que abrí `cmd` como administrador y usé el siguente comando:
<br>
<br>

```shell
bcdedit /set {current} safeboot minimal
```
{: title="Este comando se debe correr en Windows"}

Luego se debe reiniciar el compu e ir a tu `BIOS`, obviamente cada computador tiene un `BIOS` diferente, por lo que hay que buscar una opción llamada <mark>SATA Operation Mode</mark> y setearla con el valor `AHCI`.

Al guardar los cambios tu computador se reiniciará en `Modo a Prueba de Fallos` por lo que se verá un poco feo. Nuevamente hay que abrir `cmd` como Administrador y y en este caso usar el siguiente comando:

```shell
bcdedit /deletevalue {current} safeboot
```
{: title="Este comando se debe correr en Windows en Modo a Prueba de Fallos"}

{% include alert alert='En más de algún paso dirá que es necesario hacer un backup de tu computador si no quieres perder todo, obviamente eso me asustó un montón, pero la verdad es que no pasa nada y es sólo un warning por defecto.'%}

Una vez más al reiniciar e ingresar a Windows, el RST debiera estar desactivado, para chequearlo, si vas a tu `Device Manager` debieras ver algo así:

![picture of me]({{ site.urlimg }}equipo/controllers.png){: .center}

### Ahora sí a Instalar Ubuntu

Después de esto, ya se puede instalar Ubuntu, de acuerdo al video que dejé más arriba. Acá no debieran haber problemas, pero... 

> Si algo puede fallar, va a fallar<cite>Ley de Murphy</cite>

Sólo un detalle acá y es evitar instalar los drivers desde internet en la instalación.

{% include alert warning='Gracias nuevamente al blog de [David Adrian Quiñones](https://davidadrian.cc/) que adviritió de este problema. Créanme que cometí el error de instalar los drivers desde internet como mencionaban algunos tutoriales y al instalar tensorflow, mi computador colapsó y nunca más pude entrar a Ubuntu, por lo que tuve que reinstalar todo. Por lo tanto este paso es <mark>IMPORTANTE</mark>.'%}

Para solucionarlo sólo hay que preocuparse de quitar la opción de instalar third-party softwares:

![picture of me]({{ site.urlimg }}equipo/instalar.png){: .center}

{% include alert tip='Dejen una buena cantidad de `Swap Memory`, el `Swap` es un espacio del disco duro que se destinará a uso como memoria RAM en caso de que ésta se agote. Obviamente es más lenta que la memoria RAM, pero eso puede evitar que tu computador crashee. En mi caso, dejé  12GB de Swap, eso quiere decir que si llego a ocupar los 32GB de RAM, tengo aún 12GB más de margen para que el sistema siga andando. <mark>SPOILER: Realmente funciona.</mark>'%}

{% include alert success='Después de este martirio, Ubuntu debería comenzar a instalar sin problemas y no debieramos tener ningún problema más de aquí en adelante, al menos yo no lo tuve.'%}

La única preocupación que debieran tener para evitar cualquier problema es que se instalen los propietary drivers de Nvidia, de esa manera nunca más hay que usar el truco del `nomodeset`.

Para ello, hay que ir a Softwares & Updates, en la pestaña `Additional Drivers` y fijarse de NO utilizar `Nouveau`.

![picture of me]({{ site.urlimg }}equipo/drivers_ubuntu.png){: .center}

Para verificar que los drivers de Nvidia están correctos deberían poder correr esto sin errores:

```shell
nvidia-smi
```
![picture of me]({{ site.urlimg }}equipo/nvidia.png){: .center}

## PopOS

Si bien es bueno tener todos estos posibles errrores presentes para el futuro, todo lo solucioné instalando PopOS. Literalmente se instala en 5 minutos, y lo mejor de todo es que viene preinstalado con los drivers de NVIDIA, por lo que no tuve que sufrir ninguno de los problemas que sí tuve con Ubuntu. En general PopOS funciona de manera identica que Ubuntu, pero algunas mejoras las cuales pueden ver [acá](https://pop.system76.com/). Y en varios videos como este:


<div class='embed-youtube'>
{% include youtubePlayer.html id="-fltwBKsMY0" %}
</div>

<br>
Entre las características que me hicieron venirme a PopOS están el uso del Auto-tiling (que sin han usado Ubuntu se darán cuenta que es un martirio encontrar ventanas perdidas por ahí) y el uso de Shortcuts para casi todo (mover ventanas, agrandar, minimizar, mover, cambiar tamaño, envíar a escritorios virtuales, etc.). Además el estilo del escritorio es bien bonito y cuenta con un Launcher parecido a Alfred por defecto. 

## Terminal

Una de las razones por las cuales quería moverme a Linux, además de que aprovecha mucho mejor los recursos de Windows es el hecho de comenzar a acostumbrarme y perder el miedo a la línea de comandos, o terminal. Para ello busqué mucho en Youtube y obviamente también medio rayado con `Mr.Robot` traté de buscar alguna manera de que el terminal quedara bien bonito y me pudiera dar la mayor cantidad de información.

### Oh my ZSH
Lo primero que hice fue cambiarme a ZSH, la verdad es que ZSH entrega varias cosas que me permiten ser bastante más eficiente al momento de utilizar el terminal como autocompletar paths, o utilizar doble `Tab` para ver todas las carpetas dentro de la ruta, etc. Además, instalé también un framework llamado `Oh my ZSH` que básicamente trae un monton de cosas preconfiguradas que alivianan mucho la pega.

{% include alert info='Por defecto Ubuntu utiliza bash, que está bastante bien, pero la verdad es que esto lo hice más por seguir videos en Youtube, claro que ahora que lo uso, efectivamente puedo ver los beneficios.'%}

Para instalar ZSH en Ubuntu es super simple:

```shell
sudo apt install zsh
```
{: title="Instalar ZSH"}

```shell
chsh -s $(which zsh)
```
{: title="Permite dejar ZSH como el Terminal por defecto"}

{% include alert warning='Para asegurarse que ZSH quede como el shell por defecto a veces es necesario desloguearse o reiniciar el Compu.'%}

Para instalar `Oh my ZSH` es necesario tener curl o wget, la verdad creo que en mi caso utilice curl, porque en Linux algunas librerías de R piden curl. Por lo tanto utilicé ese método. Para más detalles es mejor ir al [github](https://github.com/ohmyzsh/ohmyzsh).

```shell
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
```
{: title="Instalar Oh my ZSH"}

Para terminar el proceso de Instalación también tengo un tema extra, la verdad es que no es necesario, `Oh my Zsh` ya viene con temas que son bastante atractivos, pero de nuevo Youtube me mostró un tema que lo encontré demasiado interesante, no sólo por la info que entrega si no porque es rápido y muy fácil de configurar, se trata de powerline10k.

```shell
git clone --depth=1 https://github.com/romkatv/powerlevel10k.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k
```
{: title="Descargar powerline10k"}

{% include alert warning='Es probable que tengas que instalar git si es que tu sistema no lo tiene, en ese caso, se instala así:

```shell
sudo apt install git-all
```
'%}

Para finalizar utilicé por un tiempo un emulador de terminal, que también es gracias a [David Adrian Quiñones](https://davidadrian.cc/), el que se llama Terminator.

```shell
sudo apt install terminator
```

Mi terminal queda de la siguiente manera:

![picture of me]({{ site.urlimg }}equipo/terminal.png){: .center}

### Configuración del Terminal

Entonces, luego de instalar todo, el terminal tiene que configurarse, y voy explicar cómo hacerlo:

#### Terminator 

Terminator permite una interfaz multi-terminal en una sola ventana, lo que es bastante útil. Por ejemplo en la imagen anterior, tengo 3 terminales, uno que está corriendo el servidor local de Jekyll, con el que estoy probando este artículo, y tengo dos terminales a la derecha libre. Terminator permite crear infinitos terminales, el límite es el espacio disponible para efectivamente utilizar el terminal.

Algunos comandos rápidos:

* `Ctrl+E` divide el terminal en dos de manera horizontal.
* `Ctrl+O` divide el terminal en dos de manera vertical.
* `Alt+flechas` permite moverse entre los terminales.
* `Ctrl+w` cierra el terminal activo (No la ventana completa sólo en el que estás actualmente).
* `Ctrl+x` se enfoca en el terminal activo, llevándolo a pantalla completa. Repitiendo el comando se vuelve a los terminales divididos.

#### Powerline10k
Como se puede ver, `Powerline10k` ofrece un terminal repleto de información. Para activarlo, lo primero que uno debe hacer es activarlo en el `~/.zshrc`.

{% include alert todo='Una cosa que aprendí en Ubuntu es que hay muchos archivos de configuración del tipo "~/.algo`rc`", `~` implica que estás en tu carpeta root, el `.` significa que es oculto, el `algo` es lo que estas configurando (zsh, bash, vim, etc.) y `rc` es que es el archivo de configuración. Cada vez que por ejemplo se modifique `~/.zshrc` es recomendable reiniciar la terminal para aplicar los cambios o en su defecto correr:

```shell
source ~/.zshrc
```
'%}

```shell
nano ~/.zshrc
```
{: title="Abre el archivo de Configuración de ZSH"}

Una vez abierto hay que buscar algo similar a esto y rellenar `ZSH_THEME` con el tema que nos interesa, que en este caso es `powerlevel10k/powerlevel10k` (ojo, dos veces, no todos los temas se hacen así, en este caso sí).

```shell
# Set name of the theme to load --- if set to "random", it will
# load a random theme conom/ohmyzsh/ohmyzsh/wiki/Themes
ZSH_THEME="powerlevel10k/powerlevel10k"

# Set list of themes to pick from when loading at random
# Setting this variable when ZSH_THEME=random will cause zsh to load
# a theme from this variable instead of looking in $ZSH/themes/
# If set to an empty array, this variable will have no effect.
# ZSH_THEME_RANDOM_CANDIDATES=( "robbyrussell" "agnoster" )
```
Todo el tema es completamente configurable a través de una <q>wizard</q> que sale la primera vez o mediante:

```shell
p10k configure
```
{: title="Wizard de Configuración de powerline10k"}

![picture of me]({{ site.urlimg }}equipo/terminal_2.png){: .center}

En mi caso, yo tengo:

* La ruta en la que estoy parado,
* El estado en git, el amarillo quiere decir que hay archivos que no están en stage, o que han sido modificados, mientras que el verde implicará que se acaba de hacer el commit y está todo guardado.
* Tiene la hora,
* El ambiente de conda en el que estoy,
* Y un símbolo $\checkmark$, que implica que el comando está bien, también puede haber una $\Large{✘}$ si se ingresa un comando incorrecto. Tambien puede aparecer el tiempo que demora en realizarse un comando, en verdad, es bastante útil.

{% include alert alert='Para evitar problemas de renderizado de los íconos de `powerline10k`, es necesario instalar una fuente especial, en mi caso yo instalé [MesloGS](https://github.com/romkatv/powerlevel10k-media/raw/master/MesloLGS%20NF%20Regular.ttf). Instalarla es muy sencillo, al descargarla, la abren y en la esquina superior derecha tiene la opción instalar. Además en mi caso, al tener una pantalla de extremadamente alta resolución a veces se recomienda aumentar el tamaño de la fuente para eliminar pifias del renderizado, en mi caso yo utilizo tamaño 15.'%}

#### ZSH Plugins

En este caso, ahora hay que activar plugins. En general, esto es muy sencillo gracias al archivo de configuración de Oh my ZSH. Para buscar plugins se puede ir [aquí](https://github.com/ohmyzsh/ohmyzsh/wiki/Plugins) en donde se listan todas las extensiones. 

```shell
nano ~/.zshrc
```
{: title="Abre el archivo de Configuración de ZSH"}

```shell
# Which plugins would you like to load?
# Standard plugins can be found in $ZSH/plugins/
# Custom plugins may be added to $ZSH_CUSTOM/plugins/
# Example format: plugins=(rails git textmate ruby lighthouse)
# Add wisely, as too many plugins slow down shell startup.
plugins=(git zsh-autosuggestions zsh-syntax-highlighting extract)

source $ZSH/oh-my-zsh.sh

# User configuration

# export MANPATH="/usr/local/man:$MANPATH"

```
{: title="Activar Extensiones"}

Para activar las extensiones sólo hay que buscar esa parte del archivo y agregar el nombre de la extensión y listo. Yo no uso muchas pero las explico a continuación:

![picture of me]({{ site.urlimg }}equipo/terminal_3.png){: .center}


* git: genera atajos de git, no la ocupo mucho porque se me olvidan los atajos 😛.

* zsh-autosuggestions: Me da sugerencias de qué comando puedo utilizar, para aceptar la sugerencia sólo es necesario presionar $\rightarrow$. La sugerencia se ve en gris.

* zsh-syntax-highlighting: Pinta en color los comandos para diferenciarlos, por ejemplo conda. Lo interesante es que sólo pinta comandos que estén correctos o de aplicaciones ya instaladas. Por ejemplo, si escribo `jupyter` pero no lo tengo o estoy en un ambiente conda sin Jupyter aparecerá en rojo.

* extract: Como sabrán en Linux hay varias formas de comprimir un archivo, por lo tanto, hay que saber varios comandos, extract permite utilizar un sólo comando para cualquier extensión mediante:

```shell
extract archivo.zip
extract archivo.rar
extract archivo.tar.gz
```
{: title="Descomprimir cualquier archivo"}

{% include alert warning='`zsh-autosuggestions` y `zsh-syntax-highlighting` no son extensiones estándar por lo que para su instalación es necesario descargarlas de su repo en github corriendo los siguientes comandos:

```shell
git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions
git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting
```
'%}

#### Colorls

Esto es realmente una tontera que ví en Youtube y que la verdad es bien útil para ordenar un poco el cómo se muestran tus archivos. Colorls es una gema de Ruby, por lo tanto hay que instalar Ruby.

La manera en la que yo lo hice es la siguiente:

```shell
sudo apt install ruby ruby-dev ruby-colorize
```
{: title="Descargar Ruby"}

```shell
sudo gem install colorls
```
{: title="Instalar Colorls"}

Finalmente para no tener que usar el comando `colorls` y utilizar esta propiedad sólo utilizando `ls` modificamos el archivo de configuración `~/.zshrc` y agregamos lo siguiente:

```shell
alias ls='colorls'
```
![picture of me]({{ site.urlimg }}equipo/colorls.png){: .center}


{% include alert success='Perfecto, todo lo que mostré es opcional, pero la verdad es que luego de utilizarlo uno realmente se da cuenta que la productividad aumenta montones, y obviamente me sirvió para entretenerme y perderle el miedo al Terminal en Ubuntu.'%}

## Data Science y Machine Learning

### Python

En mi caso escogí Miniconda, porque no quería descargar infinitos paquetes que nunca uso. Instalar Miniconda es muy sencillo, se descarga este [archivo](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh) y luego se instala con:

```shell
bash Miniconda3-latest-Linux-x86_64.sh
```
{: title="Instalación de Miniconda"}

{% include alert warning='Para asegurarse que `powerline10k` reconozca tu ambiente conda, hay que poner Yes a la última pregunta que aparece al instalar Miniconda. Si aún así no funciona, es mejor revisar estos links: [solución_1](https://medium.com/@sumitmenon/how-to-get-anaconda-to-work-with-oh-my-zsh-on-mac-os-x-7c1c7247d896), [solución_2](https://github.com/conda/conda/issues/8492).'%}


### R
R fue un poco más engorroso, porque habían muchos tutoriales distintos. Creo que luego de harta investigación seguí [este](https://linuxconfig.org/how-to-install-rstudio-on-ubuntu-20-04-focal-fossa-linux). 

```shell
sudo apt update
sudo apt -y install r-base gdebi-core
```
{: title="Instalar R y gdebi"}

`gdebi`, es la herramienta que permitirá instalar RStudio. Para eso hay que descargar Rstudio desde [acá](https://rstudio.com/products/rstudio/download/#download) y al menos hasta ahora, sólo está disponible una versión para Ubuntu 18, por lo que hay que elegir esa.

```shell
sudo gdebi rstudio-1.4.1103-amd64.deb
```
{: title="Instalar Rstudio"}

{% include alert success='Listo, los dos principales lenguajes usados en Data Science están listos.'%}

{% include alert alert='La verdad es que la única razón por la que terminé instalando R es porque lo necesité para la Universidad, de no haberlo necesitado no lo hubiera instalado ya que practicamente no estoy usándolo.'%}

### VS Code

Para terminar, uno de los editores que más estoy usando junto con Jupyter es VCode. La instalación es sumamente sencilla. Sólo se debe descargar el archivo `.deb` desde [acá](https://code.visualstudio.com/download) y  listo. Además se pueden descargar certificados para la actualización automática:

```shell
wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg
sudo install -o root -g root -m 644 packages.microsoft.gpg /etc/apt/trusted.gpg.d/
sudo sh -c 'echo "deb [arch=amd64,arm64,armhf signed-by=/etc/apt/trusted.gpg.d/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" > /etc/apt/sources.list.d/vscode.list'
```
{: title="Instalar certificado para actualizaciones"}

{% include alert tip='Alternativamente se puede instalar como un snap app. Pero la verdad es que no ví tan buenos reviews, ya que las snap apps son más pesadas, y más lentas, pero puede ser una opción:
```shell
sudo snap install --classic code
```
'%}


Si abren VS Code se darán cuenta que el terminal no se ve bien, esto debido nuevamente a problemas de fuentes. Para solucionar esto, es necesario instalar la fuente `MesloLGM Nerd Font` desde [acá](https://github.com/ryanoasis/nerd-fonts/releases/download/v2.0.0/Meslo.zip).

Luego en VS Code, se utiliza `Ctrl+,` para abrir la configuración y en el archivo `settings.json` hay que agregar la siguiente línea: <mark>"terminal.integrated.fontFamily": "MesloLGM Nerd Font"</mark>



{% include alert info='Cabe mencionar que llevo un tiempo intentando aprender VIM. Me gustan los atajos de teclado de VIM pero no me gusta tener que utilizar la letra `i` para comenzar a escribir. Por lo que aprovechando las capacidades avanzadas de programar macros y atajos especiales en mi teclado es que utilizo practicamente todos los atajos de VIM pero en cualquier parte del computador incluyendo VSCode.'%}

## Y listo!!!

Sé que fue un tutorial/review sumamente largo, pero aprender a instalar todo esto me tomó muchas horas de investigación y no creo poder lograrlo de nuevo, jajaja. Espero que esto sirva para ayudar a muchas personas que están intentando hacer lo mismo y que mi <q>yo del futuro</q> lo agradezca cuando ya no recuerde como hacerlo.

Espero que sirva para comenzar este 2023 motivados, porque una de las razones por las que no quiero volver a trabajar de manera presencial es porque en mi casa estoy demasiado cómodo y tengo todo lo que necesito. Incluyendo mi propio servidor. 

Ahora una de los desarrollos interesantes que generó VSCode en su última versión es el tunneling con el cual puedo en teoría acceder a cualquier computador remoto, tipo SSH, pero sin port forwading, que es una de las cosas que nunca pude terminar de configurar para poder conectarme a Jarvis fuera de mi casa. Nunca la he necesitado tampoco gracias a la pandemia, pero pronto algún tutorial con eso. 

Nos vemos a la próxima.

[**Alfonso**]({{ site.baseurl }}/contact/)

*[throttling]: Disminución del rendimiento para evitar altas temperaturas.






