# AI2 — Fase 2: preparación documental local

Base revisada: `Cotizador_AI2`, rama `enrichment`, commit
`3e24c6130074bb6bc4f7e19b59224dc1e28c1c82`.

## Alcance

Esta fase implementa una pieza reutilizable del nuevo flujo: identidad de fuentes,
páginas físicas, imágenes de referencia y caracteres nativos de PDF con coordenadas.
Se ejecuta de forma local y separada del extractor actual. No requiere API key, Backend,
Frontend ni Uvicorn. No llama a Gemini, no ejecuta OCR y no interpreta medidas ni cantidades.

`PREPARED` significa que se generaron los archivos de preparación. NO significa que
el documento esté entendido, que el texto sea completo o que la extracción esté aprobada.
La preparación todavía NO se conecta automáticamente a `GeminiExtractionProvider`.

## Archivos del paquete

Todos son NUEVOS. Combinar carpetas con las del repositorio, nunca borrarlas.

- `app/models/source_preparation_v1.py`: contrato interno de opciones y resultados.
- `app/services/source_preparation_v1.py`: servicio genérico para PDF/JPEG/PNG.
- `manual_tests/corpus_v1/prepare_sources.py`: ejecutor local del corpus registrado.
- `tests/test_source_preparation_v1.py`: 46 comprobaciones automáticas de la nueva pieza.
- `manual_tests/corpus_v1/SOURCE_PREPARATION.md`: este documento.

NO se reemplazan `gemini_extraction.py`, `semantic_review.py`, `extraction_prompt.py`,
`pyproject.toml`, `uv.lock` ni los archivos de Fase 1. Los cambios locales de diagnóstico
que todavía no has guardado quedan intactos.

## Instalación

Desde la raíz de AI2, en PowerShell y con tu entorno habitual:

```powershell
$env:UV_CACHE_DIR="$PWD\.uv-cache"
uv add "pypdfium2>=5.8,<6"
uv run pytest tests/test_corpus_v1_runner.py tests/test_source_preparation_v1.py -q
uv run pytest -q -x
```

`uv add` añade el lector/renderizador de PDF a `pyproject.toml` y actualiza `uv.lock`.
Pillow ya está declarado en el proyecto consultado. No se incluyen binarios ni fuentes
externas en este ZIP. La instalación de dependencias necesita conexión; la preparación
posterior no hace llamadas de red. No imprimir ni compartir claves.

Los tests nuevos se invocan por ruta porque el proyecto tiene una lista explícita de
`testpaths`. El tercer comando prueba por separado la suite ya registrada en el repo.
No se ha ejecutado esa suite completa en la preparación de este paquete.

## Preparar los nueve proyectos

Usar el **ZIP original** que entregaste, no el paquete de código de Fase 1/Fase 2.
Puede permanecer fuera del repositorio. No hay que subir PDFs al frontend.

```powershell
$zip = (Read-Host "Pega la ruta completa de Requerimientos.zip").Trim().Trim('"')
uv run python .\manual_tests\corpus_v1\prepare_sources.py "$zip"
```

Se comprueban nombres, tamaños y SHA-256 con el manifiesto de Fase 1. Ante un corpus
modificado se detiene; no se utiliza un archivo que sólo coincide por su nombre.

El runner descomprime cada entrada a un archivo temporal de nombre generado, la procesa
y elimina ese temporal. Nunca usa las rutas del ZIP para escribir en tu equipo. No
modifica el original. Conserva por separado el orden de entradas y la identidad estable.
El servicio de `app/` es genérico; no contiene nombres de clientes, referencias V-XX ni
valores esperados de la auditoría. El manifiesto sólo organiza/verifica los fixtures.

Para una ejecución local más corta se puede usar, por ejemplo, `--project proyecto_1`.
Eso filtra DATOS DE PRUEBA, no cambia reglas de lectura. Sin filtro prepara todos.

## Salida

Se crea una carpeta nueva por ejecución:

```text
manual_tests/corpus_v1/results/source_preparation/<run-id>/
├── inputs_verification.json
├── source_preparation_report.json
└── documents/
    └── doc-<identidad>/
        ├── source.json
        ├── page-0001.png
        ├── page-0001.json
        └── ...
```

Las imágenes independientes usan `image.png` e `image.json`, con `page_number=null`.
No se inventa una página PDF para una foto. `results/` ya está excluido de Git por Fase 1.

El reporte identifica proyecto, archivo, hash, estado y rutas de sus artefactos. Guarda
versiones de Python/motores y hashes de los archivos de código ejecutados, además de
`git_head` cuando está disponible: HEAD por sí solo no describe cambios sin commit.
No registra variables de entorno, credenciales ni el contenido de `.env`.

Los estados de error se conservan por documento/página. Una página que no pudo leerse
no se contabiliza silenciosamente como correcta. El proceso devuelve 0 cuando generó
toda la preparación, 1 si hubo documentos/páginas parciales o fallidos y 2 ante un error
de entrada, integridad o dependencia. Si se interrumpe, no usar la carpeta incompleta:
volver a ejecutar genera otra sin sobrescribir la anterior.

## Coordenadas y texto: contrato preciso

- `document_id` combina ruta lógica normalizada y SHA-256. No depende del orden.
  Archivos de iguales bytes en rutas distintas permanecen separados; `content_sha256`
  permite reconocer esa igualdad después. No se reemplazan los `source-N` de la API.
- `page_number` es el índice físico PDF (base 1). `pdf_page_label` viene de metadatos.
  `sheet_number_from_drawing` queda vacío: no se deduce una lámina del nombre del archivo.
- PDFium produce la imagen respetando la rotación intrínseca y el recorte visible.
  La conversión de las cajas de caracteres usa el mismo motor, tamaño y rotación.
- `bbox_pdf_canvas` conserva la caja nativa izquierda/inferior/derecha/superior.
  `unclipped_bbox_px` es la caja transformada sin recortar. `visible_region` es sólo su
  intersección con la página visible, normalizada y con origen superior izquierdo.
  Fuera de página o con tamaño no representable queda sin región; se informa, no se
  mueve la caja arbitrariamente para hacerla encajar.
- `characters` conserva índice nativo, texto decodificado y ubicación disponible.
  Espacios/saltos generados por PDFium se identifican como tales. No son cotas observadas.
- `raw_text_engine_order` sigue el orden del lector: NO certifica el orden de lectura
  de una tabla. No reconstruimos filas o columnas mediante adivinación.
- La presencia de caracteres nativos NO significa que todo texto visible esté disponible.
  Puede haber textos dibujados como trazos o una capa OCR previa que discrepe de la imagen.
  Se registra `NATIVE_TEXT_PRESENT_UNVERIFIED`, no una puntuación de calidad inventada.
- Bracamonte y las imágenes mantienen una representación visual aunque no se extraiga
  texto nativo. No se descartan ni se ejecuta OCR como sustitución silenciosa.
- Para imágenes se aplica únicamente EXIF Orientation cuando existe, registrándolo.
  No se intenta adivinar la orientación del croquis, corregir perspectiva o borrar tachones.
- El tamaño del lienzo PDF no es una medida de construcción. No se deducen metros/mm
  de un tamaño de página, resolución o proporción de píxeles.

## Límites y lo que sigue pendiente

La vista de página está limitada por resolución/memoria (4096 px por lado y 12 millones
de píxeles por defecto). Es contexto visual, no garantía de legibilidad de cada cota.
En la fase de localización se deberá renderizar el detalle desde el PDF original, con
su cuadro y cotas, a resolución adecuada: ampliar un PNG pequeño no recupera información.

Este paquete no identifica referencias, tablas, cotas/extremos, función, paneles ni
materiales. Tampoco etiqueta automáticamente cada región como schedule/planta/detalle.
No cambia las ocho diferencias de la baseline guardada. Las siguientes fases consumirán
estas fuentes localizadas, producirán observaciones técnicas y resolverán su significado.

La API actual conserva su soporte previo de otros formatos: este módulo nuevo sólo
prepara PDF y JPEG/PNG del corpus inicial. No se retira ni reinterpreta XLSX.

Hay límites de bytes, píxeles, páginas y caracteres; un límite alcanzado se reporta.
Son barreras locales, NO una sandbox de seguridad completa. Antes de conectar a cargas
no confiables de producción faltan aislamiento por proceso, timeout de CPU/memoria y
estrategia de workers. PDFium no permite llamadas concurrentes entre threads: el módulo
serializa sus propias llamadas; no ejecutar otros consumidores PDFium simultáneamente
en el mismo proceso sin protección compartida.

## Validación de esta entrega

- Entorno de pruebas disponible: Linux, Python 3.13.5, pypdfium2 5.8.0, Pillow 12.3.0.
  El repo requiere Python >=3.14; validar también allí. La resolución de `uv add` puede
  instalar una versión posterior dentro de la serie 5; el reporte registra cuál se usa.
- 46 pruebas nuevas + 35 del comparador = **81 pruebas aprobadas**.
- Cobertura focal: rotaciones PDF 0/90/180/270, recortes con origen desplazado, asociación
  de caracteres a píxeles con tinta, texto vertical, texto fuera de CropBox, PDF sin texto,
  8 orientaciones EXIF, transparencia, hash/orden de fuentes, errores y límites, seguridad
  de rutas ZIP, integridad del corpus y procesamiento sin conexión.
- Baseline original se mantiene en 17 checks, 9 coincidencias y 8 diferencias.
- Lectura/preparación del ZIP original: **17 documentos, 40 páginas PDF y 3 imágenes**.
  39 páginas entregan texto nativo y 1 no; no se ejecutó extracción semántica.
- No se ejecutó la suite completa de AI2, ni se validó el extractor en Windows, ni se
  certificó la exactitud de todas las transcripciones. No hay nuevas llamadas a Gemini.

## Qué compartir después de ejecutarlo

Pegar el resumen de la consola y adjuntar `source_preparation_report.json` del run.
No subir todos los PNG/JSON de caracteres salvo que necesitemos inspeccionar una página.
No hacer commit de resultados ni documentos. Después de revisar la prueba local se
agregarán selectivamente los cinco archivos nuevos, `pyproject.toml` y `uv.lock`, dejando
fuera los dos archivos de diagnóstico que siguen modificados localmente.

## Documentación técnica consultada

- https://pypdfium2.readthedocs.io/en/stable/python_api.html
- https://pypdfium2.readthedocs.io/en/stable/readme.html
- https://pdfium.googlesource.com/pdfium/+/refs/heads/main/public/fpdf_text.h
- https://pillow.readthedocs.io/en/stable/reference/ImageOps.html

La documentación vigente permite contrastar el contrato de las APIs; las pruebas de
esta entrega se realizaron con las versiones indicadas, no con todas las versiones.
