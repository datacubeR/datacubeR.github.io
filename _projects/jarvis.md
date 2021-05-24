---
permalink: /jarvis/
title: "Proyecto J.A.R.V.I.S"
subheadline: "Mi nuevo PC para Deep Learning"
teaser: "J.A.R.V.I.S era la computadora que ayudaba a Tony Stark a desarrollar su trabajo, yo también quiero armar la mía."
type: app
header: no
images:
  icon: projects/jarvis/JARVIS.png
  # icon_small: projects/color-filters/icon-mdpi.png
#   screenshots:
#     - url: 'projects/color-filters/Phone %233.jpg'
#       title: Lighting Color Filter
#     - url: 'projects/color-filters/Phone %235.jpg'
#       title: Color Matrix Color Filter with custom keyboard
#     - url: 'projects/color-filters/Tablet 7 %231.jpg'
#       title: Color Matrix Color Filter with sliders on a tablet
#     - url: 'projects/color-filters/Tablet 10 %231.jpg'
#       title: Porter-Duff Color Filter on a tablet
# links:
#   googleplay: net.twisterrob.colorfilters
---

![picture of me]({{ site.urlimg }}projects/jarvis/jarvis.jpg){: .left .show-for-large-up .hide-for-print width="500"}
![picture of me]({{ site.urlimg }}projects/jarvis/jarvis.jpg){: .center .hide-for-large-up width="250"} El proyecto J.A.R.V.I.S es una tontera que se me ocurrió. Esta super de moda poner nombre en Inteligencia Artificial así que dije ¿por qué no?

La idea de este proyecto, y gracias al premio del [Desafío Itaú Binnario](({{ site.baseurl }}/concurso/)), es poder armar un servidor con el que pueda generar prototipos rápidos en Machine/Deep Learning, participar en Kaggle, y hacer uno que otro proyecto personal. La razoón de esto es que aprendí que si bien un Laptop tiene la ventaja de la portabilidad y de tener todo a la mano, la verdad es que no es del todo cómodo porque no puedes colocar una GPU poderosa y porque, a pesar de que el mío tiene una RTX 2070, no es igual de potente que una tarjeta de PC y termina calentando el computador demasiado al tener todo compacto.

Por eso, gracias a mi amigo [Alejandro Paillaman](https://www.linkedin.com/in/alejandro-paillam%C3%A1n-olgu%C3%ADn-8a3b22138/), logré armar un computador/servidor para poder disponer de poder cuando se requiera.

![picture of me]({{ site.urlimg }}projects/jarvis/proyecto.jpg){: .right .show-for-large-up .hide-for-print width="500"}
![picture of me]({{ site.urlimg }}projects/jarvis/proyecto.jpg){: .center .hide-for-large-up width="250"}

Entonces voy a ir hablando de cada una de las partes, tratando de explicar el razonamiento desde de un punto de vista práctico y también contar algunos de los puntos a considerar. Hay muchas cosas que a priori no me fijé y que menos mal pude solucionar a tiempo para poder tener todo armado y funcionando.

No soy experto en hardware, gracias al Ale aprendí montones armando esto y a mi hermano [Sebastian Tobar](https://www.linkedin.com/in/sebasti%C3%A1n-tobar-arancibia-2a8189a5/) quien basicamente me lo armó mientras yo leía los manuales.

<br>

{% include toc.md %}

## GPU: El corazón del PC
![picture of me]({{ site.urlimg }}projects/jarvis/gpu-removebg-preview.png){: .left .show-for-large-up .hide-for-print width="500"}
![picture of me]({{ site.urlimg }}projects/jarvis/gpu-removebg-preview.png){: .center .hide-for-large-up width="250"}

La GPU no es para nada la pieza más importante al momento de armar un PC, pero en este caso era fundamental porque es un PC para hacer modelos de Machine/Deep Learning. Hace un tiempo he estado siguiendo varios youtubers Kaggle GrandMasters y todos ellos hablaban de la importancia de una buena tarjeta gráfica. 
Al momento de comprar mi Legion-7i, pude notar la diferencia de entrenar en una RTX 2070. Realmente es mucha la diferencia. Pero al empezar a trabajar con modelos más grandes el gran problema de una GPU es cuantos datos pueden procesar de manera simultánea. 

Aquí es cuando la RTX 3090 entra en juego. Es carísima, es lo más caro que he comprado y más cara que mi Laptop, pero vale completamente la pena. Yo compre la <mark>ASUS ROG STRIX RTX 3090</mark>, hasta ahora la que mejor desempeño ha entregado (normal y overclockeada, aunque no pretendo hacer OC).

![picture of me]({{ site.urlimg }}projects/jarvis/gpu2-removebg-preview.png){: .right .show-for-large-up .hide-for-print width="500"}
![picture of me]({{ site.urlimg }}projects/jarvis/gpu2-removebg-preview.png){: .center .hide-for-large-up width="250"}

* 10496 Cuda Cores.
* 936 GBps de Bandwidth.
* 328 Tensor Cores (3era Gen) (muy útil para Deep Learning) y 82 Ray Tracing Cores (útil para Renderizado).
* Posibilidad de NVLink para Multigpu.
* 24GB GDDR6X.

La gran diferencia de esta tarjeta es precisamente, la cantidad de memoria RAM disponible para cargar datos. Su gran impacto será entonces en modelos de Deep Learning, por lo que independiente de todos los procesos en paralelos que se pueden generar, y de las operaciones en tensores, permite crear Batches de entrenamiento mucho más grandes.

También puede ser de gran impacto en una nueva generación de algoritmos de NVIDIA mediante [RAPIDS cuML](https://docs.rapids.ai/api/cuml/stable/) que es un mirror de Scikit-Learn pero en GPU. Aún está muy nuevita y le queda harto por madurar, pero quizás en el futuro entrenar en GPU sea el estándar.

{% include alert success='Gracias de nuevo a Alejandro, porque como algunos sabrán hay una baja de stock mundial de chipsets y de tarjetas gráficas. Para conseguir esta lilteralmente tuve que salir corriendo al PC Factory más cercano porque por alguna razón aparecieron estas. A Chile han llegado bien pocas y la verdad es que hubo una polémica porque algunas tarjetas fallaron de mala manera por temas de drivers y de construcción y de las que salieron mejor paradas fueron la `ASUS ROG STRIX`, las `MSI` y la `Founder Edition` (que es practicamente imposible conseguir). Todas tienen leves diferencias en Reloj y ventilación, aparte de eso, son todas igual de potentes.'%}

## CPU
![picture of me]({{ site.urlimg }}projects/jarvis/ryzen-removebg-preview.png){: .left .show-for-large-up .hide-for-print width="500"}
![picture of me]({{ site.urlimg }}projects/jarvis/ryzen-removebg-preview.png){: .center .hide-for-large-up width="250"}
En este caso, me la jugué por lo que está ganando popularidad me fui por Ryzen. Le instalé un <mark>Ryzen 7 5800X</mark> de 3era generación. Es un procesador que ha tenido muy buenos reviews, porque es extremadamente potente. Tiene 8 cores y 16 threads con velocidad base de 3.8Ghz y he visto que le han sacado hasta 5GHz multicore (eso sí congelándolo), pero si no llega a los 4.7GHz.

Principalmente esto beneficia en el número de procesos paralelos y tiene directo impacto con modelos de Machine Learning de Scikit-Learn. Debido a que éstos corren en CPU más threads permite paralelizar Cross Validation, OnevsRest o algunos ensambles. El único punto a considerar es que es un procesador grande que se calienta al parecer bastante, por lo que es necesario invertir en buena refrieración.

## Motherboard

![picture of me]({{ site.urlimg }}projects/jarvis/mobo-removebg-preview.png){: .left .show-for-large-up .hide-for-print width="500"}
![picture of me]({{ site.urlimg }}projects/jarvis/mobo-removebg-preview.png){: .center .hide-for-large-up width="250"}
Acá también me la jugué por algo gama media, si bien podría haber adquirido algo superior, esta placa ha tenido excelentes reviews y está catalogada como la mejor en calidad/precio y es la <mark>TUF X570-PLUS Wifi</mark>. Estoy muy contento con esta placa, pero quizás me faltó un poco de research acá. Es tremenda placa sólo que hay cosas que no me fijé:

* Tiene Wifi5, el cuál es muy bueno y no me ha dado ningún problema, pero existe Wifi6,
* Tiene capacidad de 2 discos duros M.2, pero sólo una conexión con disipador. A mí no me molesta, tengo sólo un disco, pero si quisiera agregar otro quedará sin disipación.
* Tiene sólo una entrada PCIe reforzada, lo cual es más que suficiente, pero si quisiera otra GPU probablemente tendré problemas.
* No caben bajo ninguna circunstancia 2 RTX 3090, pero está bien. No creo tener dinero para comprar otra.
* No es compatible con Ryzen 3era Generación directamente, hay que actualizar BIOS.

Pareciera una tarjeta terrible, pero no, es tremenda tarjeta.

{% include alert warning='Si van a comprar una tarjeta madre para Ryzen de 3era generación hay que fijarse que sean 3rd-Gen Ready. En ese sentido muchas gracias a la gente de [Nice One](https://n1g.cl/Home/) quienes tienen la tarjeta a muy buen precio y me actualizaron la BIOS. No tuve ningún problema para ensemblar.'%}

Lo principal y que es algo que estoy en proceso de implementar es que es compatible con Wake-on-LAN, es decir me permite encender el PC a través de Internet. Muchos puntos por eso!!

## Refrigeración

![picture of me]({{ site.urlimg }}projects/jarvis/darkpro-removebg-preview.png){: .right .show-for-large-up .hide-for-print width="500"}
![picture of me]({{ site.urlimg }}projects/jarvis/darkpro-removebg-preview.png){: .center .hide-for-large-up width="250"}
Este apartado es el tema que menos domino, y del que menos entendía su importancia. En mi Laptop empecé a notar que siempre está muy caliente. Esto porque tiene un Intel i7 y la RTX2070. En estado base la temperatura de tarjeta de video es de 46°-48°. Por lo tanto investigando, encontré un tremendo post en el que se aconsejaba ventilación por Aire. Quizás en el mundo gamer la ventilación liquida la lleva y es mucho más llamativa, pero este tipo es Phd, haciendo modelos de Deep Learning todo el día y decía que para una GPU mejor aire, así que aire no más.

Bueno, porque estaba en oferta y porque leí buenos comentatios me fui por el <mark>be Quiet Dark Pro Rock 4</mark>. Algunos foros decían que el Noctua es mejor, pero por poquito y la verdad que conseguir un Noctua se me hizo imposible por la pandemia. Pero estoy contento, no sólo porque mi PC está extremadamente silencioso (una de las ventajas de este disipador) y porque la CPU en estado base está en 46°, que por lo que leí está bastante bien.

![picture of me]({{ site.urlimg }}projects/jarvis/fan.png){: .right .show-for-large-up .hide-for-print width="500"}
![picture of me]({{ site.urlimg }}projects/jarvis/fan.png){: .center .hide-for-large-up width="250"}
Adicionalmente tengo 4 ventiladores extra, uno que venía con el case que ayuda al Dark Pro a sacar aire, y 3 que inyectan aire frío. Aquí me fui por los <mark>Corsair AF140</mark>. Los compré porque era de los más barato que pillé, y me topé con que son top de línea y se nota, al comenzar a funcionar pude sentir la cantidad de aire que inyectan. En estado base la RTX 3090 está a 28°. Más que contento.

Un único inconveniente que no me fijé, es que los conectores para ventilador de la placa madre eran de 4 pines y estos ventiladores son de 3 pines. Afortunadamente el case traía un convertidor (creo que era del case, no lo sé la verdad) y todo funciona muy bien, pero esto es algo que no me fijé y que pudo impedir que el PC estuviera funcionando.

## Fuente de Poder

![picture of me]({{ site.urlimg }}projects/jarvis/psu-removebg-preview.png){: .right .show-for-large-up .hide-for-print width="500"}
![picture of me]({{ site.urlimg }}projects/jarvis/psu-removebg-preview.png){: .center .hide-for-large-up width="250"}
Esto era también super importante. Los requerimientos de energía de este sistema son bastante altos. Para una RTX 3090 Nvidia recomienda una fuente de al menos 750W. Varios reviews dicen que uno anda bien justo, y que se han presentado cortes al exigir a su tope a la Fuente. Además tiene que ser de calidad porque no quiero quemar todos mis componentes por abaratar costos acá. Por eso me fui por la <mark>EVGA GQ 1000</mark>, de 1000W. Si es que quisiera agregar más cosas puedo hacerlo sin miedo. Además tiene certifición GOLD y no puede venir con más cables.

<br>
<br>
<br>

## Disco Duro (SSD)

![picture of me]({{ site.urlimg }}projects/jarvis/ssd.png){: .left .show-for-large-up .hide-for-print width="500"}
![picture of me]({{ site.urlimg }}projects/jarvis/ssd.png){: .center .hide-for-large-up width="250"}
Me fui por un Samsung 970 Evo Plus. Es lo mejor que pude encontrar, tiene velocidad de escritura y lectura bien altas (3500 MB/seg y 3300 MB/seg respectivamente) y para mí es más que suficiente. El disco duro no suele ser un componente que tenga gran impacto excepto para cargar datos al momento de entrenar, en especial en los DataLoaders en Pytorch. 

<br>

## RAM

![picture of me]({{ site.urlimg }}projects/jarvis/ram.png){: .right .show-for-large-up .hide-for-print width="500"}
![picture of me]({{ site.urlimg }}projects/jarvis/ram.png){: .center .hide-for-large-up width="250"}
Acá también se recomendaba algo no muy caro, ya que la velocidad de RAM no es de gran impacto al momento de entrenar modelos. En general, dado que los modelos de Machine Learning se cargan en memoria al momento de paralelizar era importante tener al menos 32GB y las que elegí fueron las <mark>T-Force Delta</mark>. Son DD4 a 3200MHz y aparte tienen luces, que no me mataba, pero se ve bien bonito, sino miren la foto del inicio.



## Case

![picture of me]({{ site.urlimg }}projects/jarvis/case-removebg-preview.png){: .left .show-for-large-up .hide-for-print width="500"}
![picture of me]({{ site.urlimg }}projects/jarvis/case-removebg-preview.png){: .center .hide-for-large-up width="250"}
Todas estos componentes no saben en cualquier case. Por lo tanto, me recomendaron utilizar un full tower, y me fijé que fuera muy grande, que cupiera todo sin problemas y que permitiera la entrada de mucho aire. Por eso escogí el <mark>Cougar Panzer Max-G</mark>. Dentro de las características que me gustaron es que trae todo lo que necesitas (muchos tornillos, piezas de ensamblaje y sobre todo el adaptador para los ventiladores que me salvó la vida), tiene vidrio templado para ver los componentes con RGB, permite controlar la intensidad de los ventiladores, permite como hasta 8 ventiladores (si tengo problema de temperatura tiene mucho espacio para seguir refrigerando) tiene 4 USBs adicionales, conector de micrófono, audífonos y un gancho para colgarlos.
Además permite un ensamblado super ordenado, trae compartimento aislado para la Fuente de Poder y separación para la gestión de cables, además se siente muy muy firme.

## Sobre el ensamblado

La verdad es que no es muy difícil. Al momento de comprar hay que fijarse en verificar que todo sea compatible. Para eso utilicé una página que se llama [PCpartpicker](https://pcpartpicker.com/) que la encontré luego de ver un video de Abishek Thakur en la cual explicaba en qué fijarse para armar un buen PC para Machine Learning.

Para el tema de compras y encontrar todas las partes en Chile 🇨🇱 utilicé [Solotodo](https://www.solotodo.cl/). Tremenda página, salen todas las partes y donde conseguir stock de ello, además de calificaciones de cada tienda. Yo compré en casi todas las que tenían buena calificación y realmente muy buen servicio.

Los principales obstáculos que me encontré fueron la BIOS de la placa madre (lo cual fue solucionado gentilmente por la gente de [Nice One](https://n1g.cl/Home/)), y los conectores de 3 pines. Quizás lo más difícil fue la gestión de cables pero ahí el Seba se encargó de dejarme el PC muy muy ordenado.

### ¿Donde compré?

* [Winpy](https://www.winpy.cl/): Compré el Case y CPU. Son un poco enredados para los presupuestos, pero lo que compré me llegó al otro día. 
* [Nice One](https://n1g.cl/Home/): Ellos son una casita acá en Viña. Pero me atendieron super bien, compré el SSD, RAM y la placa y salió todo muy bueno. Además que me regalaron stickers y por sobre todo actualizaron la BIOS.
* [killstore](https://www.killstore.cl/): Acá fue el único lugar que encontré ventiladores de 140mm. No se dedican a vender partes, al parecer son una tienda de diseño gráfico, pero estaban muy baratos y también llegaron al otro día.
* [SP Digital](https://www.spdigital.cl/): Acá compré Fuente de Poder y Dark Pro. El Dark Pro estaba muuuuy barato, la Fuente, muuuy cara, pero era el único lugar con fuentes de 1000 y de calidad. Se demoraba 10 días hábiles el envío pero terminó demorando 3, muy bueno.
* [PC Factory](https://www.pcfactory.cl/): Acá compré la GPU. Caro, muy caro, pero es el único lugar donde llegó esta tarjeta. Me dejaron reservarla (llegaron sólo 4) y me dieron todas las facilidades para pagarla así que muy agradecido.

### Y ahora qué?

Ahora me queda nada más que probar, ya instalé Ubuntu siguiendo las instrucciones que dejé de cómo setear tu [Equipo]({{ site.baseurl }}/equipo/). Debido a que esta maquina es sólo Ubuntu, es decir, no hice Dual Boot, decidí usar la opción <q>...Borrar todo e instalar Ubuntu...</q> y ya (la verdad no lo decidí, fue la única manera de que me funcionara). Esto porque al hacer dual Boot es Windows quien genera el booteo del computador. Investigando noté que si uno no tiene Windows uno tiene que crear una partición que se encargue de eso. Al usar la opción descrita arriba Ubuntu lo hace automático.

Además mi idea es poder acceder a este computador/servidor desde donde sea, para ello voy a mostrar un pequeño [tutorial]({{ site.baseurl }}/blog/2021/02/ssh) de cómo conectarse utilizando SSH, y cómo me conecto de manera remota. Lo último que me faltaría es probar el Wake-on-Lan que permite activar mi PC remotamente para encenderlo y luego acceder por SSH.



*[OC]: Overclocking: Aumentar la velocidad de fábrica para mejorar desempeño.
*[SSH]: Protocolo de traspaso de datos seguro.
