# AI2 — Fase 3: localización documental verificable

Base de integración consultada: `Cotizador_AI2`, rama `enrichment`, commit
`4a61f54efd4a933369a7e5f50ca6b1b25e274c5a`.

## Alcance de esta entrega

Añade siete archivos; NO reemplaza código anterior, no cambia dependencias, endpoints,
contrato AI2/Backend, prompts existentes ni `pyproject.toml`.

- `app/models/document_localization_v1.py`
- `app/services/document_localization_v1.py`
- `app/services/localization_prompt_v1.py`
- `app/providers/gemini_localization_v1.py`
- `manual_tests/corpus_v1/localize_sources.py`
- `manual_tests/corpus_v1/LOCALIZATION.md`
- `tests/test_document_localization_v1.py`

Usa los resultados de Fase 2, sin repetir la preparación ni volver a subir el ZIP por
el frontend. Trabaja con la carpeta COMPLETA de preparación, no solamente su reporte.
El servicio genérico no lee `expected_checks.json`, `acceptance_matrix.json`, nombres
especiales de proyectos ni valores esperados. El runner conserva project_id para no
mezclar proyectos, no para aplicar reglas distintas.

Esta fase propone dónde están los elementos y su evidencia. Todavía NO resuelve
width/height, cantidad comercial, operación, componentes físicos, vidrio ni precios.
Una propuesta de región no prueba que contenga las cotas o el cuadro correctos.
La comprobación visual y la cobertura contra la anotación del corpus siguen pendientes.

## Modos separados

### 1. `plan`: local y sin red

Valida identidades, páginas físicas, coordenadas de los artefactos, tamaños y existencia
de PNG/observaciones de Fase 2. Congela sus hashes, el prompt y el esquema de respuesta.
Lista una solicitud de localización por vista: en este corpus, 40 páginas PDF + 3 JPG.
Las fotografías mantienen page_number=null. Una página sin texto nativo NO se omite.

Para vistas cuyo lado mayor supera 2400 px, `--tiles auto` añade cuatro ampliaciones
solapadas a la imagen completa, dentro de LA MISMA solicitud. No son elementos nuevos.
El modo por defecto es auto; `--tiles none` conserva solo la imagen completa.

El plan valida archivos y marcos; la codificación de las imágenes para la API y su
presupuesto de bytes se comprueban al ejecutar las solicitudes seleccionadas, antes
de crear el cliente. No se reserva crédito ni se paga nada al planificar.

### 2. `run`: opt-in con Gemini, puede tener costo

Requiere `--allow-paid-calls` y un presupuesto `--max-calls` explícito. Si la selección
requiere más llamadas, rechaza todo antes de crear el cliente. No se recortan trabajos
silenciosamente para ajustarse al presupuesto. No hay reintentos automáticos.

Cada solicitud envía UNA página/imagen y sus ampliaciones; no reenvía todos los PDF.
Adjunta tokens nativos limitados y ubicados como ayuda NO VERIFICADA; no un supuesto
texto completo en orden de lectura. Las truncaciones se señalan explícitamente.

El proveedor usa `gemini_api_key` y `gemini_model` de los Settings existentes, sin cambiar
el modelo configurado. Importa Settings/SDK solamente al ejecutar este modo. Las claves
no se escriben en los informes ni se imprimen excepciones completas del SDK.

Las vistas fotográficas PNG de Fase 2 se re-codifican para el modelo en JPEG calidad 95,
sin cambiar tamaño, por el presupuesto de bytes. Eso es una conversión con pérdida, no
una mejora de resolución. Se registra el tipo y hash de cada imagen enviada. Las
fuentes, los PNG de Fase 2 y los recortes de auditoría permanecen sin esa re-codificación.
Los dibujos PDF se envían como PNG. El tamaño inline tiene un límite conservador.

Las respuestas completas se guardan antes de validarlas, sin reparar coordenadas,
identidades o JSON truncado. Una respuesta que no terminó en STOP se registra como fallo.
En el primer error de API/esquema/escritura se detienen nuevas llamadas y se marcan los
trabajos restantes como no ejecutados. Cada página produce un informe independiente.

Las cajas usan [y0,x0,y1,x1] en escala 0..1000 del preview COMPLETO. El almacenamiento
incluye también x/y/width/height normalizados 0..1 y los límites efectivos en píxeles.
Las relaciones entre dibujo, tabla, rótulo, cotas y notas quedan como PROPOSED/AMBIGUOUS.
No se suman cantidades, no se eliminan candidatos repetidos y no se propagan notas.
Candidatos sin referencia formal se conservan con reference_raw=null.

### 3. `replay`: volver a validar una respuesta guardada, sin Gemini

Permite comprobar recortes/validadores sobre la misma respuesta. No vuelve a generar
la lectura del modelo. Si cambian el prompt/esquema o los artefactos, se rechaza un plan
incompatible; no se atribuyen respuestas antiguas a imágenes distintas.

## Instalación y comprobación inicial

Combinar app/, manual_tests/ y tests/ del paquete con las carpetas del repositorio.
No borrar carpetas. Ningún archivo anterior requiere sobrescribirse.

Desde la raíz de AI2, en Git Bash:

```bash
export UV_CACHE_DIR="$PWD/.uv-cache"
uv run pytest tests/test_corpus_v1_runner.py tests/test_source_preparation_v1.py tests/test_document_localization_v1.py -q
uv run pytest -q -x
```

En PowerShell, solamente cambia la primera línea por:

```powershell
$env:UV_CACHE_DIR="$PWD\.uv-cache"
```

Los tests nuevos se nombran explícitamente porque la configuración actual enumera
archivos de test. Esta entrega no cambia esa configuración.

Planificar usando TU reporte y conservando su carpeta documents/:

```bash
uv run python manual_tests/corpus_v1/localize_sources.py plan --prepared "manual_tests/corpus_v1/results/source_preparation/b03a4c7b11314d3cadd1ec479a16b41d/source_preparation_report.json"
```

El identificador anterior es el run que compartiste. Para otra preparación, sustituye
la ruta por su source_preparation_report.json. No copiar el reporte solo a otro lugar.

Salidas nuevas, dentro del directorio results/ ya ignorado por Git:

```text
manual_tests/corpus_v1/results/source_localization/plan-<id>/
  localization_plan.json
  localization_plan_report.json
```

El segundo archivo es el resumen que puedes compartir para revisar la selección.
PLANNED_REQUESTS y PLANNED_IMAGES NO son llamadas ni imágenes ya enviadas.

## Ejecución real posterior, después de revisar el plan

No ejecutar por accidente durante la instalación. Usa la API key existente en esa misma
consola o en tu .env local. No la compartas ni la imprimas.

Ejemplo de selección de un proyecto desde cualquier shell (una sola línea):

```bash
uv run python manual_tests/corpus_v1/localize_sources.py run --plan "RUTA/AL/localization_plan.json" --project proyecto_1 --max-calls 3 --allow-paid-calls
```

El proyecto es solo una selección de prueba, no una regla del extractor. Para procesar
el corpus completo de este plan, omitir --project y fijar conscientemente --max-calls 43.
También se puede seleccionar --document D03; el servicio usa la identidad real del
archivo y su vista, no el nombre D03 como señal de contenido.

Conservar el mismo plan/modelo para comparar ejecuciones. Aun con temperatura 0 no se
certifica estabilidad ni exactitud. Esta fase todavía no compara todos los elementos
con el gold; el comparador de Fase 1 no acepta este nuevo esquema intermedio.

Artefactos de cada run:

```text
run-<id>/
  run_started.json
  localization_report.json
  <job-id>/
    request_prompt.txt
    response_envelope.json
    job_report.json
    proposals/
      localization.json
      regions_overview.png
      r1.png ...
```

Los recortes son de la imagen preparada a su resolución disponible, NO un render nuevo
más nítido del PDF original. La relectura de detalle desde el PDF original será una pieza
posterior cuando los recortes evidencien falta de legibilidad. Las vistas completas y
las notas no se eliminan. Una rotación sugerida se registra, no se aplica automáticamente.

Las transcripciones del modelo se etiquetan MODEL_TRANSCRIPTION_UNVERIFIED. Los índices
nativos de caracteres dentro/intersectando una región son soporte geométrico, NO prueba
de literalidad ni de pertenencia al elemento. No se hereda confianza de un rectángulo.

Ejemplo de replay sobre un response_envelope.json guardado:

```bash
uv run python manual_tests/corpus_v1/localize_sources.py replay --plan "RUTA/AL/localization_plan.json" --job "ID-DEL-JOB" --response "RUTA/AL/response_envelope.json"
```

## Validación realizada al preparar esta entrega

- Entorno local de pruebas: Linux, Python 3.13.5, Pydantic 2.13.4, Pillow 12.3.0.
- 147 pruebas pasaron y 1 se omitió en los tres archivos de tests combinados.
- De la Fase 3: 66 pasaron; 1 prueba de configuración del SDK se omitió porque
  google-genai no está instalado en el host de pruebas. Las respuestas de las pruebas
  de ejecución son simuladas, y los sockets están bloqueados en los tests nuevos.
- En el entorno del proyecto, donde google-genai está instalado, esa prueba comprueba
  la construcción del request con el SDK real usando una respuesta simulada, sin red.
- Se probaron imágenes sintéticas, cajas inválidas, relaciones colgantes, claves JSON
  duplicadas, identidad/orden de fuentes, rutas, límites, pérdida de metadatos, ausencia
  de texto, candidaturas repetidas/sin referencia, fallos, replay y protección de fuentes.
- Se generaron artefactos de Fase 2 sobre el ZIP original para probar el plan sobre sus
  17 documentos/43 vistas. Se ensamblaron solicitudes de las 43 vistas en grupos locales
  y se verificaron presupuesto de bytes, formatos e identidades sin enviarlas a Gemini.
- El plan local dio 9 proyectos, 17 documentos, 43 vistas, 43 solicitudes planificadas,
  183 imágenes planificadas y 4 vistas sin tokens nativos disponibles (PDF visual + fotos).
- El baseline anterior conserva 17 comprobaciones, 9 coincidencias y 8 diferencias.
- NO se ejecutó el modelo real, NO se midió precisión de localización y NO se ejecutó
  la suite completa de la aplicación. Confirmar tests y plan en Windows/Python 3.14.

## Decisión de avance

No aprobar por número de cajas o tests verdes. La siguiente validación visual debe
comprobar referencia, cuadro, cotas, pertenencia, cobertura y regiones omitidas en las
nueve familias. Solo después se conectará esta evidencia al resolvedor de campos.
Este paquete no elimina el flujo actual ni entrega otra cotización al Backend.

## Referencias técnicas utilizadas

- Google Gen AI SDK: https://googleapis.github.io/python-genai/
- Entrada visual: https://ai.google.dev/gemini-api/docs/image-understanding
- Salida estructurada: https://ai.google.dev/gemini-api/docs/structured-output

El esquema válido no demuestra contenido semánticamente correcto. La API real y el
modelo configurado deben validarse en una ejecución opt-in. No se modificó ningún
repositorio remoto, ni se hizo commit/push desde esta entrega.
