---
permalink: /vscode/ 
title: "Mis Extensiones para Trabajar en VSCode"
subheadline: "Preparando un IDE para trabajar en Ciencia de Datos"
teaser: "Por qué prefiero VScode por sobre otros entornos."
# layout: page-fullwidth
usemathjax: true
category: ds
header: no
image:
    thumb: vscode/vscode.png
tags:
- tutorial
published: true
---

![picture of me]({{ site.urlimg }}vscode/vscode.png){: .left .show-for-large-up .hide-for-print width="500"}
![picture of me]({{ site.urlimg }}vscode/vscode.png){: .center .hide-for-large-up width="500"}

Trabajando como Científico de Datos, o de cualquier programador en general, pasas gran parte de tu día en frente de algún editor de código o un IDE. Si bien yo realmente creo que no está totalmente definido cuál es el mejor ambiente para trabajar, a mí me gusta la versatilidad que ofrece VSCode.  <!--more-->. 

En el mundo de los datos uno trabaja en varios frentes: Desde exploración o diseño de prototipos en Notebooks, hasta diseño de pipelines automatizados utilizando Scripts, y desplegando sistemas en Cloud o tecnologías asociadas como Docker. En general uno tiene que manejar muchas tecnologías en distintos entornos. Y a pesar de que me gustan las tecnologías especialistas, tener un entorno multifacético como VSCode me gusta mucho, en especial porque es un todo en uno: Python Scripts, Jupyter Notebooks, terminal además de permitir conectarme a entornos remotos como contendores de Docker o mi servidor utilizando SSH. Todo en un sólo lugar.

{% include alert warning='La intención de este artículo no es juzgar qué es mejor y donde trabajar. Yo tengo mi opinión y me gusta trabajar en VSCode. Básicamente quiero compartir y documentar para mi `yo del futuro` cuál es mi configuración elegida.' %}

Lo bueno de VScode es que es completamente personalizable, pero me pasa que siempre que veo videos/artículos que abordan las mejores extensiones están enfocadas en un público más de desarrollo Web, o en otros lenguajes de programación más que en Python en el contexto de Ciencia de Datos. La idea de este artículo es mostrar extensiones conocidas y otras bien `under` que me han permitido aumentar la productividad (uno de mis sueños que es no depender tanto del mouse, ni tampoco de Vim/NeoVim que no me terminan de convencer) o sencillamente que tu ambiente de trabajo se vea más bonito. 

{% include alert info='Como editor de Texto en verdad creo que `Vim/NeoVim` es lo mejor que hay por velocidad, atajos, funcionalidades, y otras capacidades como IDE. Donde no me termina de convencer es como un entorno de ejecución en tiempo real de Python (tipo REPL). A diferencia del desarrollo de software uno necesita ir ejecutando trozos de código en memoria rápidamente y la verdad es que no he encontrado algo que me termine de gustar en `VIM`. El `Jupyter Notebook` es lo mejor (o menos malo) que he encontrado, pero tiene muchos otros problemas. Para mí la mejor solución la tenía Atom con Hydrogen, pero sabrán que Microsoft decidió darlo de baja y eliminarlo (igual tenía algunas pifias de performance, se volvía muy lento con archivos con muchas líneas de código, pero la idea de tener un Script con el output inmediatemente al lado es para mí lo mejor que hay. VSCode ofrece el Python Interactive Window pero no anda ni cerca de lo útil que era Hydrogen, de hecho funciona muy mal). Me encantaría que algo así saliera en VSCode pero al parecer es medio imposible por la forma en la que VSCode funciona.'%}

## Python

![picture of me]({{ site.urlimg }}vscode/python.png){: .center width="700"}

Obvio, esta es probablemente la primera extensión a instalar para poder ejecutar código en Python. No mucho que decir, permite ejecutar Python en VSCode, permite la interactive Window, que es para ejecutar código en vivo tipo REPL pero en Python Script, además de habilitar los Jupyter Notebooks (antes eran extensiones separadas pero ahora viven todo bajo la extensión de Python). Actualmente esta extensión igual instala extensiones amigas como Jupyter Cell Tags, Jupyter Keymap, etc. Pero son extensiones que trabajan <q>behind the scenes</q>. 

La extensión además permite refactoring (aunque no funciona mucho), e incorpora Pylance como Language Server para dar sugerencias de código, y isort como una herramienta para ordenar imports. Además tiene incorporación con formatters (yo uso Black) y linter (la verdad es que odio los linters por la cantidad de suciedad que agrega a mi pantalla, pero normalmente trabajo con flake8 pero ahora estoy probando Ruff).

Una de las cosas que más me gusta es que por fín integraron la posibilidad de modificar Tags para poder usar librerías súper interesantes como Papermill. 

{% include alert tip='Entiendo que aún existe gente que usa R, y la verdad es que si bien VSCode tiene una extensión para trabajar con R, creo que RStudio funciona mucho mejor. Si les interesa hablar de la pelea R vs Python, lo dejamos para otra ocasión.'%}



# Extensiones de estética

Probablemente muchos pueden pensar que no son necesarias pero de verdad hacen que nuestro ambiente de trabajo sea más ameno. Al menos el hightlighting creo que ayuda mucho a leer mejor el código y poner atención a distintas partes del código. El resto me hace más feliz!! Pero igual ayuda.


## One Dark Pro

![picture of me]({{ site.urlimg }}vscode/one_dark.png){: .center width="500"}

No mucho que decir acá más que es mi tema favorito (que de hecho proviene de Atom). Para mí es bien importante que el código sea muy multicolor, y que permita diferenciar cada parte del código. He probado muchos temas y este por lejos es el mejor, tiene colores predefinidos para: 

* Palabras Claves
* Funciones
* Clases
* Métodos
* Atributos
* Strings
* Números
* Booleanos
* Signos

Una imagen de cómo se ve código Python con casi todo lo que mencioné se puede ver acá:

![picture of me]({{ site.urlimg }}vscode/sneak_peak.png){: .center width="700"}

## Material Icon

![picture of me]({{ site.urlimg }}vscode/material_icon.png){: .center width="500"}

Esta es una extensión muy sencilla que coloca íconos a las carpetas y los íconos de manera muy bonito para reconocerlos mejor. Utiliza el nombre o la extensión para asignar íconos que facilitan encontrar archivos y diferencias carpetas importantes de otras no tan importantes.

![picture of me]({{ site.urlimg }}vscode/icon.png){: .center width="300"}

{% include alert info='Una cosa que me gusta bastante de VSCode es que marca qué archivos no están siendo trackeados por Git, cuáles son nuevos, cuáles son partes de `.gitignore`, etc. Si bien no es ninguna extensión adicional lo que permite esto, Material Icon si agrega esos colores dorado, verde, o gris para denotar distintos estados de un archivo en Git.'%}

## indent-rainbow

![picture of me]({{ site.urlimg }}vscode/indent-rainbow.png){: .center width="500"}

Es una extensión muy pequeñita que permite marcar la indentación con colores para asegurarse que estén alineados. Si está incorrectamente alineado aparecerá en rojo, sino irán marcando por colores los distintos niveles.

![picture of me]({{ site.urlimg }}vscode/indent.png){: .center width="500"}

## Better Comments

![picture of me]({{ site.urlimg }}vscode/better_comments.png){: .center width="500"}

Esta extensión permite destacar ciertos comentarios. En general lo he encontrado bien útil para poder tener mensajes a los que tengo que estar atento en el futuro o para que mis compañeros vean mis mensajes. Básicamente, tiene varios tipos de mensajes que se destacarán dependiendo del símbolo con el que partan:

![picture of me]({{ site.urlimg }}vscode/comments.png){: .center width="300"}

Si bien se sugieren esos comentarios, la verdad es que uno puede usar el color para lo que uno quiera.

## Rainbow CSV

![picture of me]({{ site.urlimg }}vscode/rainbow_csv.png){: .center width="500"}

Esta es otra pequeña extensión muy livianita, que permite colorear un CSV para poder tener una mejor lectura del archivo raw. Es la extensión más común, y si bien hay otras extensiones que permiten ver los CSV cómo Spreadsheets, a mí me gusta esta, porque de nuevo, me gusta el editor bien multicolor.

![picture of me]({{ site.urlimg }}vscode/csv.png){: .center width="700"}


# Autocompletado

Son extensiones que ayudan a autocompletar código o escribir código de manera más rápida y oprimiendo menos teclas.

## Github Copilot

![picture of me]({{ site.urlimg }}vscode/copilot.png){: .center width="500"}

La verdad es que cada vez lo ocupo menos. Si bien es un gusto que me complete mucho código, rara vez me da la respuesta absolutamente correcta, lo que significa que tengo que terminar editando la sugerencia. Lo mejor que tiene es sugerir variables de manera correcta o librerías que necesito importar, pero normalmente editar para mí es más lento que escribir todo. Una de las razones por las que he dejado de usarlo es porque hago clases donde la mayor parte del tiempo hago código en vivo, y no es ninguna gracia que te sugieran largas líneas de código. 

Sirve, vale la pena (yo lo tengo gratis por ser estudiante), pero no es la gran maravilla. Podría vivir sin él y prefiero algo como Intellicode que permita autocompletar más rápido y entender mejor lo que estoy haciendo.


## Intellicode

![picture of me]({{ site.urlimg }}vscode/intellicode.png){: .center width="500"}

Esta extensión ha terminado siendo mucho más útil. Intellicode es bien interesante, ya que permite mejorar las capacidades de Pylance para dar mejores sugerencias. No sé si habrán dado cuenta de que muchas veces las sugerencias son bien malas. Uno siempre importa csv y al colocar `pd.read_` se sugiere `.read_clipboard()` porque alfabéticamente va primero. Al habilitar Intellicode, el motor de sugerencias va aprendiendo para entregar mejores sugerencias de acuerdo a tu código y a tus prácticas de escritura. Verás que mejores sugerencias se ven con una estrella:

![picture of me]({{ site.urlimg }}vscode/sugerencia.png){: .center width="300"}

{% include alert info='Solía haber una extensión llamada Kite que era muy similar y que la verdad funcionaba sumamente bien hasta que comenzar a cobrar por todas las cosas buenas que tenían. Lamentablemente quebró y encontré esta que ha suplido bastante bien mis necesidades.'%}


## Path Intellisense

![picture of me]({{ site.urlimg }}vscode/path_intellisense.png){: .center width="500"}

Esta extensión permite el autocompletado de los paths/rutas reconociendo que archivos están disponibles a medida que uno construye la ruta. Sumamente útil, en especial para los que tenemos mala memoria recordando donde guardamos nuestra info. Creo que lo único que no me gusta es que exige para disparar las sugerencias (al menos inicialmente) el partir el path con `./`. A parte de eso, muy buena extensión.

![picture of me]({{ site.urlimg }}vscode/path.png){: .center width="500"}

## Python Type Hint

![picture of me]({{ site.urlimg }}vscode/type_hint.png){: .center width="500"}

Como el nombre lo dice, sugiere autocompletado para cuando quieres utilizar Type Hints. 

![picture of me]({{ site.urlimg }}vscode/hint.png){: .center width="300"}

# Superpoderes

Extensiones que permiten que haga cosas de manera más rápido o eficiente normalmente utilizando sólo el teclado.

## Autodocstring

![picture of me]({{ site.urlimg }}vscode/autodocstring.png){: .center width="500"}

Es otra pequeña extensión que entrega el template de un Docstring. Para los que no sepan, un Docstring es una documentación propia de una función en Python. Lo bueno de esto es que no sólo entrega una pauta de referencia de cómo rellenarla sino que además VSCode la renderiza de manera muy bonita cuando se hace hover sobre la función con el mouse. 

Es importante recalcar que VSCode soporta muchos formatos de docstring, pero yo ocupo estilo `Numpy`, el cuál se puede configurar en `Settings > AutoDocstring:Docstring Format`.

La extensión automáticamente identificará los parámetros de entrada y si es que existe un `return` y mostrará placeholders para reemplazar el tipo de dato y una descripción de cada elemento.

![picture of me]({{ site.urlimg }}vscode/docstring.png){: .center width="400"}

## Quick and Simple Text Selection

![picture of me]({{ site.urlimg }}vscode/quick_simple.png){: .center width="500"}


Probablemente la mejor extensión para los que nos gusta usar el teclado y sumamente desconocida (pueden ver el número de descargas). Tiene atajos de teclados que permiten seleccionar todo lo que se encuentre entre cualquier tipo de paréntesis o comillas. Además tiene otra funcionalidad que permite cambiar el tipo de comillas inmediatamente.

En mi caso lo tengo configurado como acorde, es decir, presionados como secuencia, no al mismo tiempo, esto gracias a mi Corne:
* `Ctrl k ;`  seleccionará todo lo que está dentro de cualquier tipo de comillas. 
* `Ctrl k a` seleccionará entre paréntesis, `Ctrl k s` entre corchetes y `Ctrl k d` entre llaves. La elección de esto tiene que ver con cómo hago los distintos de paréntesis en mi teclado.
* `Ctrl k :` irá rotando entre "", '' y ``. Súper útil.

## Advanced New File

![picture of me]({{ site.urlimg }}vscode/advanced_new_file.png){: .center width="500"}

Es una extensión que sólo permite crear un archivo nuevo. La gran gracia es que puedo especificar su ruta completa y creará carpetas intermedias que se requieran para su creación en el caso que no existan. Me gusta principalmente porque permite agregar un atajo de teclado para hacerlo más rápido.

![picture of me]({{ site.urlimg }}vscode/new_file.png){: .center width="500"}


## TabOut

![picture of me]({{ site.urlimg }}vscode/tabout.png){: .center width="500"}

Probablemente la extensión más simple pero más útil del mundo. Una de las ventajas que tiene VSCode es el auto-cerrado de paréntesis y comillas, que es genial para que nunca olvides cerrarlos. El problema que trae eso es que luego tienes que usar la flecha a la derecha ➡️ para salir del paréntesis o cierre de comillas. Esto es particularmente un problema porque la flecha a la derecha suele estar lejos en los teclados convencionales y eso significa perder la posición de home row si haces touch typing (que es una demora innecesaria). Esta extensión tiene el único objetivo de usar la tecla Tab para salir de un cierre de paréntesis o comillas. Demasiado simple, pero no les puedo explicar lo productivo que es. 


## Python postfix completion

![picture of me]({{ site.urlimg }}vscode/postfix.png){: .center width="500"}

Todavía no me termino de acostumbrar a esta extensión, pero tiene varias cosas muy interesantes. Por ejemplo: Si tengo un objeto en Python llamado `item` y hago `item.len` se transforma automáticamente en `len(item)`. Esto siempre pasa, no te das cuenta que es una lista y no tiene `.shape` y tienes que devolverte a hacer el `len()`. Tiene varios atajos más para return, for loops, if statements, y funciones de conversión como `int()`.

## Python Indent
![picture of me]({{ site.urlimg }}vscode/python_indent.png){: .center width="500"}

Por alguna razón VSCode al presionar Enter dentro de alguna estructura de datos o alguna sintaxis que requiera indentación, no mantiene la indentación. Esta extensión asegura que si se mantenga. Simple! Por ejemplo si escribo un `else:` y presiono Enter automáticamente la siguiente línea está indentada.




# Misceláneo

Todas las extensiones que no supe como clasificar.

## Git Graph

![picture of me]({{ site.urlimg }}vscode/git_graph.png){: .center width="500"}

Otra extensión muy pequeñita. Muchos recomiendan el uso de Git Lens (pero realmente no entiendo para qué sirve). Esto lo único que hace, pero lo hace muy bien, es mostrar de manera mucho más bonito un `git log`.

![picture of me]({{ site.urlimg }}vscode/graph.png){: .center width="300"}

La extensión muestra todas las ramas, como interactúan entre ellas y cada uno de los commits. Simple y bonito!!

## Markdown All in One

![picture of me]({{ site.urlimg }}vscode/markdown.png){: .center width="500"}


Como el sitio lo llevo principalmente en Markdown, uso esta extensión para habilitar atajos como `Ctrl + B` para Negrita o `Ctrl + I` para Cursiva, además de funcionalidades para trabajar mejor en Markdown como previsualizar o evitar el autocompletado innecesario.

## Remote SSH

![picture of me]({{ site.urlimg }}vscode/ssh.png){: .center width="500"}

La mejor extensión que existe para conectarte de manera remota a un servidor. La verdad es que por ahora ser cliente Movistar me dio un pequeño problema porque no tenía autorización al puerto 22 que es el puerto por defecto para usar SSH, pero aparte de eso, requiere cero configuración y me permite conectarme de manera remota a mi servidor JARVIS.

## Project Manager

![picture of me]({{ site.urlimg }}vscode/project_manager.png){: .center width="500"}

Creo que todavía no le termino de sacar el beneficio a esta extensión, pero básicamente, en vez de tener que abrir cada carpeta de proyecto, o navegar por el terminal dependiendo de donde quieres trabajar te coloca una pestaña en la que tienes todos tus proyectos. Por lo que basta con abrir VSCode donde sea y ahí tienes todo. No me termino de acostumbrar porque tengo la costumbre de entrar siempre por terminal. Pero es una excelente extensión.

## DVC

![picture of me]({{ site.urlimg }}vscode/dvc.png){: .center width="500"}

Esta es una extensión para poder generar experimentos en DVC, muy similar a lo que sería Tensorboard. Probablemente muy útil (no la he usado mucho aún), pero requiere de un tutorial por sí sola.

{% include alert tip='La verdad es que tengo más extensiones instaladas, pero que en verdad no les he encontrado el uso. Una de ellas es `Bookmarks`, nunca la he usado. Hay otras que me parecen bien interesantes como `Docker` o `Dev Containers`, pero me gusta mucho el uso del terminal, por lo que casi siempre termino interactuando mediante él.' %}

Otra cosa que para mí es sumamente importante es el terminal. Si bien no es una extensión adicional, sí utilizo mucho el terminal integrado para no tener que abrir ventanas adicionales. 

![picture of me]({{ site.urlimg }}vscode/terminal.png){: .center width="900"}

Mi terminal utiliza `Oh my ZSH` con `Powerlevek10k`. Esto me permite tener mucha información en la línea del terminal como:

* La ruta actual en la que estoy parado.
* La rama de Git además archivos en stage, si commits esperando push o incluso un stash. Además me indica estados como rebase y el estado del rebase en caso de conflictos.
* El ambiente de Python en el que estoy actualmente.
* Hora.
* Me indica si el estado del comando fue exitoso o no.

Y tiene autocompletado, syntax highlightning entre otros. Vale mucho la pena.

Ahora, para que todo esto valga la pena tiene que ir de la mano con la configuración de VSCode, y por sobre todo los atajos de teclados. Yo diría que utilizo la mayoría de atajos por defecto, pero tengo algunos customizados que hacen mi vida más sencilla (en combinación con la disposición de algunas teclas en mi teclado como `Ctrl`).

Pero lo dejamos para la otra,

Espero que les haya gustado y les sirva para armar un ambiente más ameno y cambiarse a VSCode.


[**Alfonso**]({{ site.baseurl }}/contact/)
