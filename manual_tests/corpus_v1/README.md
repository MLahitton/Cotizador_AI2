# AI2 — Fase 1: base de pruebas del corpus

## Objetivo

Registrar el corpus y comenzar a medir errores con entradas congeladas. Este paquete
no cambia la extracción ni pretende resolver las medidas: permite comparar resultados
sin pagar nuevas llamadas a Gemini para cada cambio determinístico.

Repositorio de destino: `Cotizador_AI2`, rama de trabajo `enrichment`.
No modifica `app/`, Backend, Frontend, prompts, dependencias ni configuración de pytest.
Todos los archivos de código/configuración de este paquete son nuevos.

## Qué contiene y qué NO certifica

- `manifest.json`: 9 proyectos y 17 documentos con nombres y SHA-256 del ZIP original.
  El inventario auditado registra 14 PDF con 40 páginas y 3 JPG.
- `acceptance_matrix.json`: los 33 casos descriptivos de la auditoría previa, conservados
  como PROPUESTOS/NO EJECUTADOS. Abarcan los nueve proyectos. No son 33 tests automáticos.
- `expected_checks.json`: primera anotación parcial de valores: 16 checks de campos de
  V-01, V-15 y V-19, más un check del inventario V-01 a V-19 de PROYECTO 1. No es el gold
  completo de los 17 documentos. Los otros ocho proyectos figuran PENDING.
- `runner.py`: verificador de integridad de ZIP y comparador offline de snapshots.
- `tests/test_corpus_v1_runner.py`: pruebas del comparador y del manifiesto, sin importar
  la aplicación ni inicializar proveedores. No certifican que Gemini extraiga bien.
- `local_only/`: copia privada del último snapshot `(3)` recibido en el chat, su discovery,
  límites de procedencia y tres imágenes de evidencia utilizadas en los checks iniciales.
  Está excluido de Git por `.gitignore`. No contiene el ZIP original ni claves.
- `results/`: salida local de los comandos; también excluida de Git.

Las referencias concretas son DATOS DE PRUEBA; nunca se utilizan para corregir producción.
`MATCH` significa coincidencia del check, NO aprobación de evidencia, estado o documento.
Los valores no anotados, los conflictos y las familias no evaluadas NO cuentan como éxitos.

## Instalar

1. Descomprimir el paquete fuera del repositorio.
2. Copiar `manual_tests/corpus_v1/` y `tests/test_corpus_v1_runner.py` en esas rutas del repo.
3. No reemplazar archivos de `app/`. No copiar el ZIP de documentos a una carpeta versionada.
4. Conservar los cambios locales de diagnóstico anteriores; este paquete no los modifica.

En PowerShell, desde la raíz de AI2 y con su entorno habitual activo:

```powershell
$env:UV_CACHE_DIR="$PWD\.uv-cache"
uv run pytest tests/test_corpus_v1_runner.py -q
uv run python .\manual_tests\corpus_v1\runner.py baseline
```

El primer comando debe comprobar la herramienta. El segundo NO inicia AI2, Uvicorn ni
una extracción: compara el snapshot guardado con las expectativas iniciales.

El `pyproject.toml` consultado enumera los archivos de tests explícitamente. Por eso
el test nuevo se invoca con su ruta: no se presupone que `pytest -q` lo descubra.
Para comprobar aparte la suite existente: `uv run pytest -q -x`.

## Resultado de referencia obtenido en la preparación de este paquete

Se ejecutaron 35 pruebas propias del paquete con Python 3.13.5 y pytest 9.0.2.
No se ejecutó la suite completa de AI2 ni se importaron sus dependencias. El proyecto
consultado requiere Python >=3.14; confirmar las pruebas también en el entorno del repo.

Verificación del ZIP original: 17/17 hashes coincidentes y 9 proyectos registrados.

Sobre el snapshot guardado `02-enrichment(3).json`:

```text
STATUS=PARTIAL_BASELINE
CHECKED_VALUES=17
MATCHING_VALUES=9
NONMATCHING_VALUES=8
CORPUS_APPROVED=NO
EXTRACTION_RUN=NO
NETWORK_CALLS=0
```

Las ocho diferencias iniciales son: V-01 alto/operación; V-15 alto/cantidad/conteo de
paneles/geometría; V-19 cantidad/operación. Son errores o ausencias del snapshot guardado,
no regresiones causadas por esta herramienta. El resto del documento NO se da por aprobado.

El reporte completo se escribe en `manual_tests/corpus_v1/results/baseline.json`.
`baseline` devuelve código 0 si pudo generar el informe, aunque haya diferencias: comprobar
siempre sus contadores. `evaluate --strict` devuelve 1 ante diferencias o falta de anotación.

## Verificar el corpus sin ejecutar el extractor

```powershell
uv run python .\manual_tests\corpus_v1\runner.py verify-inputs "C:\ruta\Requerimientos.zip"
```

Sustituir únicamente la ruta. No descomprime archivos. Verifica todos los nombres y bytes
esperados, y detecta archivos ausentes/adicionales o modificados. Rechaza rutas inseguras
y duplicadas. La normalización Unicode sólo afecta a la comparación de nombres.

## Evaluar otra salida ya existente

```powershell
uv run python .\manual_tests\corpus_v1\runner.py evaluate `
  --project proyecto_1 `
  --snapshot ".\runtime\extraction-stages\ID-DEL-RUN\02-enrichment.json" `
  --stage enrichment `
  --out ".\manual_tests\corpus_v1\results\run_nuevo.json" `
  --strict
```

No hace falta volver a subir PDFs para probar un cambio del comparador. Cambios reales de
lectura/localización sí requerirán después ejecuciones controladas sobre los originales.

Soporta sólo `GeminiEnrichmentResult` (02) y `GeminiExtraction` pre-mapper (04), no el RAW
final de Backend/RequirementExtraction. Para 04 usar `--stage pre-mapper`.
No reinterpreta sistemas, no rellena datos ni modifica el snapshot.

Las medidas se comparan numéricamente después de convertir unidades declaradas m/cm/mm;
una unidad ausente o desconocida queda UNCOMPARABLE. No se redondean diferencias de diseño.
Se aceptan sinónimos de operación definidos explícitamente en los tests. Una referencia
repetida no se selecciona por orden: queda IDENTITY_AMBIGUOUS hasta que el caso tenga una
clave de ocurrencia más precisa. No usar esta primera clave simple para calificar todas
las variantes repetidas del resto del corpus.

Los estados/confianza se muestran cuando están en la medida, pero no se certifican con
estos checks numéricos. La literalidad, procedencia y localización siguen por evaluar.

## Procedencia y límites

El snapshot no aporta hash de entradas, commit ni configuración exacta del modelo.
Su proyecto se declara, no se certifica mediante huellas del run. `provenance.json` lo
marca explícitamente. Verificar el ZIP por separado NO demuestra que ese run leyó esos
mismos bytes. La próxima instrumentación deberá registrar estos metadatos sin claves.

Las evidencias para los checks están en las imágenes del paquete y la auditoría anterior:
D12 = D-16 VENTANAS.pdf; D14 = D-18 VENTANAS.pdf; página física 1 en ambos casos.
Las dimensiones se expresan en metros conforme a la interpretación de la auditoría;
no se afirma que todos los rótulos numéricos lleven sufijo de unidad.

## Guardar en Git

Revisar primero `git status --short` y `git diff --cached --name-only` para no incluir
cambios ajenos ya preparados. Agregar selectivamente:

```powershell
git add manual_tests/corpus_v1 tests/test_corpus_v1_runner.py
git diff --cached --name-only
```

`local_only/` y `results/` no deben aparecer. No usar `git add -f` para esos datos.
Cuando el staging sólo incluya lo deseado:

```powershell
git commit -m "Add offline corpus baseline and validation scaffolding"
git push origin HEAD
```

No se ha ejecutado ningún commit/push desde este paquete.

## Siguiente cambio funcional

Mantener los 33 criterios como guía de capacidades, completar anotaciones sin convertir
las ambigüedades de fuente en cifras inventadas e introducir un paquete documental con
identidad, página física/plancha, orientación y evidencia localizada. El flujo anterior
permanece disponible para comparación. Después se incorporan observaciones/resolución y
se extiende el evaluador a conflictos, componentes, casos repetidos y salida Backend.

No aprobar por media global ni por reducir artificialmente candidatos a revisión. La
meta es recuperar lo resoluble y conservar pendientes legítimos; este paquete sólo
establece cómo empezamos a medirlo.
