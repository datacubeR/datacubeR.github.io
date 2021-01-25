---
permalink: /work/
title: "Mi tiempo como Profesional"
breadcrumbs: false
---

Hola soy Alfonso, entré a estudiar Ingeniería Comercial por error, me cambié a Civil por miedo a no encontrar pega, pero terminé trabajando como Data Scientist por casualidad y me encanta. Si te interesa saber de mi trayectoria profesional podrás ver en detalle lo que he hecho.

<span class="icon-install"></span> Si te interesa descargar mi [CV en español]({{ site.data.links.cv-es }}){:target="_blank"} o en [Inglés]({{ site.data.links.cv-en }}).  

Además, hace poco pude cumplir uno de mis sueños que es enseñar. Agradezco a Desafío Latam por la oportunidad y la confianza.

<section class="timeline clearfix ir">
	<h2 class="timeline-date">El futuro</h2>
	{% assign lastYear = 0 %}
	{% for job in site.jobs reversed %}{% unless job.hidden %}
		<a href="{{ site.baseurl }}{{ job.url }}" title="Click for more details">
		<article class="timeline-box {% if job.type == 'Company' %}left{% else %}right{% endif %}">
			<h3>{{ job.title }}</h3>
			<span class="icon-calendar">
				{% assign job_from = job.dates.from | date: '%Y' %}
				{% assign job_to = job.dates.to | date: '%Y' %}
				{% if job_from != job_to %}
				<span title="{{ job.dates.from }}">{{ job_from }}</span> &ndash; <span title="{{ job.dates.to }}">{{ job_to }}</span>
				{% else %}
				<span title="{{ job.dates.from }} &ndash; {{ job.dates.to }}">{{ job_from }}</span>
				{% endif %}
				{% if job_from != nil and job_to != nil %}
					(~{% include datediff.liq begin=job.dates.from end=job.dates.to measure='dynamic' %}{{ result }}&nbsp;{{ measure }})
				{% else %}
					current
				{% endif %}
			</span><br/>
			<strong>Rol</strong>: {{ job.role }}<br/>
			<strong>Sector</strong>: {{ job.sector }}<br/>
			<strong>Tecnologías</strong>: {{ job.maintech }}
		</article></a>

		{% assign currentYear = job.dates.from | date: '%Y' %}
		{% assign diff = currentYear | minus: lastYear %}{% if diff < 0 %}{% assign diff = 0 | minus: diff %}{% endif %}
		{% if forloop.first == false and lastYear <> currentYear and diff > 1 %}
			<h2 class="timeline-date">{{currentYear}}</h2>
			{% assign lastYear = currentYear %}
		{% endif %}
	{% endunless %}{% endfor %}
</section>

{%comment%}
{% for job in site.jobs reversed %}{% unless job.hidden %}
 * <span title="{{ job.dates.from }}">{{ job.dates.from | date: '%Y' }}</span>
   --
   <span title="{{ job.dates.to }}">{{ job.dates.to | date: '%Y' }}</span>
   {% include datediff.liq begin=job.dates.from end=job.dates.to measure='dynamic' %}
   {% if result != 0 %}(~{{ result }}&nbsp;{{ measure }}){% endif %}:
   [**{{ job.title }}**]({{ site.baseurl }}{{ job.url }}){: title="{{ job.type }}: {{ job.role }} in {{ job.maintech }}"}{% endunless %}{% endfor %}
{%endcomment%}
