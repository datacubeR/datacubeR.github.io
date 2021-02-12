---
permalink: /equipo/
layout: page
title: "Mi Equipo"
subheadline: ""
teaser: "Este es el equipo que utilizo para hacer Data Science"
header:
  image_fullwidth: equip.jpg
---
<!-- ...and learn more at the same time. You can message me at the [contact page]({{ site.baseurl }}/contact/). -->

{% comment %} Including the two pictures is a hack, because there's no way use the Foundation Grid (needs a container with row) {% endcomment %}
![picture of me]({{ site.urlimg }}widget2-equipo_2.jpeg){: .left .show-for-large-up .hide-for-print width="500"}
![picture of me]({{ site.urlimg }}widget2-equipo_2.jpeg){: .center .hide-for-large-up width="500"}

Para hacer ciencia de Datos hoy en día es necesario tener un equipo que te acompañe y que tenga algunos requerimientos mínimos. Y los tutoriales muestran que  todo es <mark>color de rosa</mark>, pero la verdad no es tan así, por lo tanto, decidí hacer este tutorial, mostrando las características de mi equipo, pero también todos los procesos para instalarlos y obviamente todos los errores no esperados que aparecen a lo largo de la instalción y el setup de todo. Realmente me costó mucho dejar todo funcionando, pero desde ahí en adelante mi computador jamás ha dado problemas, incluso cuando le exijo al máximo.

Entonces, ¿Cuáles son los aspectos más importantes al momento de comprar un computador? La verdad es que esta es una pregunta bien dificil de responder, pero en general, se puede resumir en alto poder de cómputo. ¿Qué es lo que en particular yo estaba buscando? En mi caso, quería un computador con características gamer, no porque juegue, sino porque están mejor preparados para una alta demanda de recursos, en partícular, la ventilación. Dado que ocupar muchos recursos generalmente genera un aumento en la temperatura es que este aspecto es bastante importante. Además me interesaba mucho la portabilidad, no busco un ultrabook, pero sí poder moverme, y es por eso que decidí que un Laptop era la mejor opción.

{% include toc.md %}

## Computador

Decidí ir por un Laptop Lenovo, la verdad es que si bien he leído algunos comentarios de algunos problemas en torno a su temperatura, yo no he tenido problemas. Debo decir que se me hizo muy complicado ver qué Laptop comprar, primero, porque no hay tanto para elegir en Chile, y segundo, porque muchos de los reviews en Youtube muchas veces son muy enfocados en gaming y terminan siendo muy críticos, por lo que siempre queda la inseguridad si será un buen computador. Finalmente hay que arriesgarse.

Ahora, cuáles son las características de mi laptop:

![picture of me]({{ site.urlimg }}equipo/neo.png){: .center}

* Legion-7i de 15.6": Principalmente me gustan los notebooks grandes y ojalá con teclado numérico. No es tan portable, en el sentido que tiene un cargador gigante y debe pesar unos 2.5 kilos en total, pero a mí eso no me molesta.

* 32GB de RAM @ 2666MHz: Partí con 16GB y en la mitad de la competencia Binnario me quedé corto y tuve que comprar más, 32GB es lo máximo que soporta el Legion 7i.

* Procesador Intel i7-10750H @5.00GHz: Aquí lo principal que hay que fijarse es en la letra final del procesador. H, es por High Performance, por lo tanto casi no sufre throttling. En el computador de mi pega, tenía un procesador terminado en U, que son la serie de ahorro de energía. El problema de estos procesadores es que siempre hacen throttling para ahorrar más energía lo cual no es algo deseado cuando se quiere utilizar el compu a máxima capacidad.

* GPU Nvidia RTX 2070 Max-Q: Esto fue un capricho, quería una Tarjeta que tuviera tensor cores para ver si se sentía la diferencia, y la verdad es que sí se siente.

* Disco Duro NVme 512GB: Acá me conformé con lo que había, 512GB para mí es más que suficiente, pero me preocupé que el disco duro fuera NVme para tener mejor performance lectura-escritura.


Creo que esa son las características principales que uno debiera mirar al momento de elegir un computador, en términos de poder computacional anda bastante bien, lo que siempre me ha tenido preocupado es el tema de la ventilación.

{% include alert alert='Para sacarle el mayor provecho al laptop siempre debe estar en una temperatura adecuada para evitar el throttling. El throttling es un mecanismo de autocuidado que puede tener tanto el procesador como la GPU para bajar su rendimiento con el fin de disminuir su temperatura. En mi caso la temperatura de mi GPU que es lo que más he recargado no ha pasado de los <mark>60°C</mark>. Es una temperatura alta, pero al menos la refrigeración hace su trabajo y nunca he sentido que me voy a quemar o algo por el estilo. Al menos la RTX 2070 tiene una temperatura máxima antes de throttling de <mark>93°C</mark>, por lo que aún tengo bastante margen.' %}

Repito, <mark>no he tenido problemas de temperatura</mark> pero siempre es algo que me asusta en especial cuando quiero dejarlo máxima potencia. La ventilación que tiene es altamente criticada en foros, la cual es un vapor chamber, pero digan lo que digan, cumple.

![picture of me]({{ site.urlimg }}equipo/vapor.png){: .left .show-for-large-up .hide-for-print width="500"}
![picture of me]({{ site.urlimg }}equipo/vapor.png){: .center .hide-for-large-up width="500"}

Hay una característica en particular que me gustó que es que tiene distintos modos que permiten un mayor nivel de ventilación. Modo Performance, cambia el color del botón de encendido a rojo y genera ventiladores a toda potencia, a mí no me molesta, porque prefiero que haga la pega de enfríar el PC (y aparte uso audífonos).
Color blanco, es media potencia, ni fu ni fa, y color azul, baja potencia, probablente útil para ahorrar energía. 

{% include alert todo='En mi caso el 95% del tiempo está enchufado y en modo performance 😀'%}

Otro aspecto que me gustó mucho es que tiene infinitas conexiones: 4 USB, 2 USB-C (1 con thunderbolt), conexión HDMI, y conexión a carga rápida, en realidad carga tan rápido como se descarga. Diría que el gran problema es que consume mucha batería y nunca he durado más de una hora sin conectar al enchufe, gran problema para algunos, pero es algo con lo que estoy dispuesto a lidiar.

> Probablemente mi caso no es el más recomendable, yo decidí ir por un laptop gamer, no es la mejor opción, pero primeramente estaba pensando en portabilidad. Debido a que ya tomé una decisión en dedicarme de manera full al Machine/Deep Learning es que pretendo invertir en un PC de escritorio con más poder, pero mientras no lo tenga, esto es lo que hay.

### Teclado

Para el caso de mi teclado quise hacer un apartado adicional, ya que es bastante especial. Este es un teclado mecánico que compré por Kickstarter que de verdad anda muy bien. Es el Epomaker GK68XS y su gracia es que tiene algunas características especiales:

![picture of me]({{ site.urlimg }}equipo/teclado.jpg){: .center}

Primero es un teclado 65% por lo que es más compacto y puedo alcanzar casi todas las teclas desde el <q>home row</q>. Tiene luces, lo cual ayuda en la oscuridad, pero no las uso mucho. Puede conectarse por USB tipo C o por Bluetooth lo cual es bastante cómodo, tiene Cherry Red switches lo cual lo hace muy agradable al tipeo, pero lo que más me gusta es que es 100% configurable. Como ven tiene 3 barras espaciadoras lo cual me permite teclas extras que yo puedo elegir, además de atajos multimedia, 3 capas de teclas y creación de macros, lo cual es muy útil para poder programar.

### El resto

No mucho más que agregar, utilizo un mouse vertical marca Zelotes, me encanta, muy útil, y tiene algunas teclas extras que me permiten navegar más rápido. Y un monitor adicional marca Samsung 21.5". Hoy por hoy se me está haciendo pequeño pero cumple con todo lo que necesito en términos de cálidad de imágen y algunos filtro de luz que me permiten estar frente al compu todo el día. También tengo unos audífonos bluetooth marca Soundcore que andan muy bien, un poquito de carga y duran muchas pero muchas horas.

Para armar el setup tengo 2 brazos hidráulicos que me permiten levantar tanto el monitor como el laptop a mi voluntad, si me da por trabajar de pie podría hacerlo (aunque no lo hago nunca). Y la razón por la que nunca lo hago es por mi silla. Una Nitro Concepts S300. Me salió muy cara, me costó mucho que llegara, no es el color que más me gusta pero desde que la compré casi no tengo dolor de espalda (muy rara vez me molesta pero porque no me estoy apoyando correctamente), y en verdad paso 10 hrs diarias en el PC, a veces más. Las características más importantes son:

* Pistón clase 4, y armazón de acero creo que soporta hasta 135 kgs.
* Reclinable full y muy blandita, no sirve de mucho, pero es rico hasta dormir acá.
* Apoya brazos 3D, podría ser 4D, lo extrañé, porque está pensada para gente muy grande, y yo no soy tanto.
* No es de cuerina, es de tela, por lo que no transpiro nada, pero... se le pega el pelo de gato, pero no se puede pedir todo. 
* Pero por sobre todo, calidad alemana, está muy bien termindada, como se une la tela al asiento, el tipo de tela, las costuras. Mis gatos la han atacado y ha resistido muy bien.

Ya, de vuelta al compu.

## Sistema Operativo

He sido una persona que he utilizado Windows toda la vida, y la verdad es que nunca me habia quejado hasta que comencé a hacer modelos más grandes, y me dí cuenta el desperdicio de recursos de Windows, además que al consumir toda la RAM el computador se cuelga y no vuelve más, eso si es que no recibes alguna pantallita azul.

Es por eso que decidí jugármela e instalé Ubuntu, en mi caso 20.04 LTS.

{% include alert tip='Acá hay un abanico de posibilidades y sabores diferentes de Linux, decidí cambiarme por algunas recomendaciones que leí en la web y porque en verdad todos decían que me haría la vida más fácil en términos de compatibilidad. Pero NO, muchos problemas ocurrieron.'%}

Luego de seguir las recomendaciones de [David Adrian Quiñones](https://davidadrian.cc/), me decidí por Ubuntu y varios otras de las sugerencias que hace (No todas).

{% include alert tip='En mi caso hice dual boot, como se explica en el video, esto implica dejar Windows instalado en caso de necesitarlo. Creo que nunca he tenido que usarlo hasta ahora, pero siempre puede salvar de un apuro.'%}

<div class='embed-youtube'>
{% include youtubePlayer.html id="-iSAyiicyQY" %}
</div>

{% include alert alert='Acá comienzan los problemas, si bien, el video muestra que instalar Ubuntu es una maravilla y no hay ningún problema asociado, la verdad es que no es así. Acá les muestro varios inconvenientes que tuve al momento de instalar Ubuntu.'%}

### No se reconocen los drivers de Video

{% include alert info='Yo ya había recién comprado un Lenovo Legion 5 que tuve que devolver porque tuve problemas con los drivers, y además un slot de RAM estaba dañado por lo que en todo momento pensé lo peor.'%}

Al comenzar con el Booteable, lo primero que veo luego del Menú de instalar Ubuntu es lo siguiente:


![picture of me]({{ site.urlimg }}equipo/nvidia_fail.jpeg){: .right .show-for-large-up .hide-for-print width="500"}
![picture of me]({{ site.urlimg }}equipo/nvidia_fail.jpeg){: .center .hide-for-large-up width="500"}

Obviamente entré en panico, y lo primero que hice fue googlear acerca de este problema. Ahí encontré que esto era algo <q>normal</q> cuando se tiene una GPU Nvidia y era sencillo de solucionar, pero era mi primera experiencia con Ubuntu y con trabajo en Terminal, por lo que siempre da como cosa modificar elementos por línea de comando.

La solución según los foros era desactivar los drivers por defecto, que normalmente son drivers open source que se llaman algo así como `Nouveau` y que no funcionan.

Para hacer eso hay que hacer lo siguiente: Justo en la pantallita que pide instalar Ubuntu hay que presionar la tecla `E`. Esto nos llevará a la siguiente pantalla:

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

Para ello, hay que ir a Softwares & Updates, en la pestaña `Additional Drivers` y fijarse de no utilizar `Nouveau`.

![picture of me]({{ site.urlimg }}equipo/drivers_ubuntu.png){: .center}

Para verificar que los drivers de Nvidia están correctos deberían poder correr esto sin errores:

```shell
nvidia-smi
```
![picture of me]({{ site.urlimg }}equipo/nvidia.png){: .center}


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

Para finalizar utilizo un emulador de terminal, que también es gracias a [David Adrian Quiñones](https://davidadrian.cc/), el que se llama Terminator.

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

### VS Code

Para terminar, uno de los editores que más estoy usando junto con Jupyter es VS Code. La instalación es sumamente sencilla. Sólo se debe descargar el archivo `.deb` desde [acá](https://code.visualstudio.com/download) y  listo. Además se pueden descargar certificados para la actualización automática:

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

## Y listo!!!

Sé que fue un tutorial largo, pero aprender a instalar todo esto me tomó muchas horas de investigación y no creo poder lograrlo de nuevo, jajaja. Espero que esto sirva para ayudar a muchas personas que están intentando hacer lo mismo y que mi yo del futuro lo agradezca cuando ya no recuerde como hacerlo.

[**Alfonso**]({{ site.baseurl }}/contact/)

*[throttling]: Disminución del rendimiento para evitar altas temperaturas.






