from pathlib import Path

ELEMENT_DISCOVERY_PROMPT = """
Ejecuta PASS 1: ELEMENT DISCOVERY para AI2.

Devuelve exclusivamente un objeto JSON valido. No incluyas markdown, comentarios ni texto
fuera del JSON.

Estructura JSON esperada:
{
  "elements": [{
    "temporary_id": string|null,
    "reference": string|null,
    "name": string|null,
    "category_raw": string|null,
    "source_ids": [string],
    "source_hint": string|null,
    "status": "explicit"|"inferred"|"ambiguous"|"unknown"|"not_applicable"|null,
    "confidence": number|null
  }],
  "notes": [string]
}

Objetivo unico de esta pasada: cobertura exhaustiva de elementos potencialmente
cotizables. No extraigas detalle tecnico en esta fase.

Instrucciones:
- Analiza TODAS las fuentes entregadas como un unico requerimiento.
- Identifica TODOS los elementos potencialmente cotizables.
- Prioriza cobertura exhaustiva sobre detalle.
- NO te detengas despues de encontrar algunos ejemplos.
- Recorre integralmente todas las paginas, imagenes, tablas, cuadros, notas y dibujos.
- Conserva todas las referencias distintas realmente encontradas.
- Usa source_ids para indicar las fuentes disponibles donde aparece o se relaciona el
  discovery. Solo puedes usar IDs listados en AVAILABLE SOURCES.
- Si no hay evidencia suficiente para asignar source_ids, devuelve source_ids=[] y
  conserva source_hint como texto auxiliar.
- No conviertas rangos como "V-01 al V-30" en referencias inexistentes.
- No inventes elementos.
- No descartes elementos incompletos.
- No agrupes elementos diferentes solo porque compartan sistema o medidas.
- Referencias repetidas pueden representar ocurrencias y deben conservarse de forma
  identificable usando temporary_id y source_hint.
- Si un elemento no tiene referencia clara, conservalo con reference=null y un
  temporary_id descriptivo.
- Excluye espejos.
- No calcules precios.
- No apliques Classic, Essentials ni otras reglas comerciales.
- No consultes catalogo.
- No infieras sistemas comerciales.

No incluyas en discovery: medidas detalladas, vidrio, perfiles, materiales, finish,
accesorios, componentes detallados, variantes detalladas, ocurrencias detalladas,
evidencia estructurada, relaciones, conflictos, precios ni catalogo.
""".strip()

ELEMENT_SCOPE_PROMPT = """
Ejecuta PASS 1.5: SCOPE INTELLIGENCE para AI2.

Devuelve exclusivamente un objeto JSON valido. No incluyas markdown, comentarios ni texto
fuera del JSON.

Recibes discoveries YA EXISTENTES. Clasifica cada discovery; no vuelvas a hacer discovery
general. Devuelve exactamente un resultado por cada temporary_id recibido.

Estructura JSON esperada:
{
  "elements": [{
    "temporary_id": string,
    "scope": "in_scope_full"|"in_scope_partial"|"out_of_scope"|"uncertain",
    "reason": string|null,
    "in_scope_components": [string],
    "out_of_scope_components": [string],
    "evidence_source_ids": [string],
    "evidence_notes": [string],
    "confidence": number|null
  }],
  "warnings": [string]
}

Definicion de alcance:
- in_scope_full: hay evidencia suficiente de que el elemento completo corresponde al
  alcance de vidrieria.
- in_scope_partial: hay evidencia suficiente de que una parte del conjunto pertenece al
  alcance de vidrieria y otra parte corresponde a otros oficios.
- out_of_scope: hay evidencia suficiente de que el elemento no contiene participacion
  real de vidrieria cotizable.
- uncertain: no existe evidencia suficiente para confirmar ni descartar de forma segura
  una participacion de vidrieria.

Contexto de alcance:
- Steel and Glass trabaja principalmente componentes de vidrieria y cerramientos
  asociados.
- El alcance puede incluir componentes de vidrio integrados dentro de conjuntos
  arquitectonicos mayores cuyo nombre principal no sea "vidrio".
- La ausencia de una especificacion explicita de vidrio en el mismo detalle no es
  suficiente, por si sola, para afirmar que el conjunto esta fuera del alcance.
- Cuando un elemento pueda razonablemente contener una participacion de vidrieria, pero
  la evidencia disponible no permita confirmarlo ni descartarlo de forma segura,
  clasificalo como uncertain.
- Prioriza evitar falsos negativos: es preferible mantener un elemento dudoso para
  revision posterior que descartarlo incorrectamente.

Instrucciones:
- Usa todas las fuentes disponibles.
- Usa evidence_source_ids para indicar las fuentes disponibles que respaldan la
  clasificacion. Solo puedes usar IDs listados en AVAILABLE SOURCES.
- Si no hay fuente estructurada segura, devuelve evidence_source_ids=[] y conserva la
  explicacion en reason/evidence_notes.
- No inventes vidrio ni componentes.
- No excluyas por nombre solamente.
- OUT_OF_SCOPE REQUIERE EVIDENCIA POSITIVA DE EXCLUSION.
- La mera ausencia de evidencia de vidrio NO equivale a evidencia de ausencia de
  vidrio.
- La ausencia de la palabra vidrio no es suficiente para excluir.
- La presencia de metal no es suficiente para incluir.
- Si existe vidrio cotizable como parte de un conjunto mayor, usa in_scope_partial.
- Si toda la solucion pertenece a vidrieria, usa in_scope_full.
- Si hay evidencia suficiente de que no contiene participacion de vidrieria cotizable,
  usa out_of_scope.
- Si no hay evidencia suficiente para confirmar ni descartar participacion de
  vidrieria, usa uncertain.
- Espejos siguen fuera del alcance V1.
- No calcules precios.
- No consultes catalogo.
- No apliques Classic, Essentials ni otras reglas comerciales.
- No produzcas recomendaciones comerciales.
""".strip()

ELEMENT_ENRICHMENT_PROMPT = """
Ejecuta PASS 2: TECHNICAL ENRICHMENT para AI2.

Devuelve exclusivamente un objeto JSON valido. No incluyas markdown, comentarios ni texto
fuera del JSON.

Estas enriqueciendo una lista YA DESCUBIERTA. NO hagas discovery libre como objetivo
principal. Devuelve exactamente un resultado por cada temporary_id recibido.

Estructura JSON esperada:
{
  "elements": [{
    "temporary_id": string,
    "reference": string|null,
    "name": string|null,
    "category_raw": string|null,
    "description": string|null,
    "quantity": string|number|boolean|null,
    "functional_type_raw": string|null,
    "operation_raw": string|null,
    "panel_count": number|null,
    "movable_panel_count": number|null,
    "fixed_panel_count": number|null,
    "modulation_raw": string|null,
    "opening_direction_raw": string|null,
    "special_features": [string],
    "measurements": [{"type": string|null, "raw_label": string|null,
      "value": number|null, "unit": string|null, "text": string|null,
      "status": status|null, "confidence": number|null, "notes": string|null}],
    "geometry_type_raw": string|null,
    "geometry_raw": string|null,
    "configuration_raw": string|null,
    "glass": [glass_item],
    "materials": [named_item],
    "profiles": [named_item],
    "finish_raw": string|null,
    "accessories": [named_item],
    "components": [component_item],
    "occurrence_context": string|null,
    "variant_context": string|null,
    "evidence": [{"source_id": string|null, "text": string|null,
      "page_number": number|null, "sheet_name": string|null, "cell_range": string|null,
      "region": {"x": number, "y": number, "width": number, "height": number}|null,
      "visual_description": string|null, "notes": string|null}],
    "evidence_notes": [string],
    "missing_or_unknown": [string],
    "status": status|null,
    "confidence": number|null,
    "notes": string|null
  }],
  "warnings": [string]
}

status: "explicit", "inferred", "ambiguous", "unknown", "not_applicable".
named_item: name, type, code, role, description, quantity, status, confidence, notes.
glass_item: type, thickness, thickness_value, thickness_unit, color, treatment,
composition, description, status, confidence, notes.
component_item: name, type, role, description, quantity, measurements, geometry_raw,
configuration_raw, glass, materials, profiles, finish_raw, accessories, status,
confidence, notes.

Instrucciones:
- NO omitas ninguno de los discoveries recibidos.
- Analiza todas las fuentes para obtener informacion del discovery correspondiente.
- Cuando una afirmacion tenga trazabilidad clara, agregala en evidence con source_id,
  text, page_number, sheet_name, cell_range, region o visual_description segun aplique.
- region usa coordenadas normalizadas relativas a la imagen/pagina visible:
  x, y, width y height deben estar entre 0 y 1; x/y son esquina superior izquierda;
  width/height son tamano del recorte, no coordenadas finales; x + width <= 1 y
  y + height <= 1.
- No uses pixeles, puntos PDF, porcentajes 0-100, grillas 0-1000 ni [x1,y1,x2,y2]
  dentro de region. Si solo conoces coordenadas absolutas o no puedes normalizarlas con
  certeza, deja region=null y conserva text/visual_description/page_number.
- Para evidencia visual de PDF, incluye page_number real y region normalizada cuando exista.
- Para evidencia visual de imagen, incluye region normalizada cuando exista y no inventes
  page_number.
- Usa exclusivamente los source_id listados en AVAILABLE SOURCES; no inventes source_id.
- Cuando una afirmacion provenga de mas de una fuente, puede indicar varias evidencias.
- MantÃ©n separadas afirmaciones contradictorias entre fuentes; no fusiones evidencia
  incompatible silenciosamente.
- Si no puedes asociar una evidencia a una fuente de forma segura, conserva el texto en
  evidence_notes y no inventes source_id.
- Relaciona informacion entre paginas, tablas y dibujos cuando pertenezca a ese item.
- No inventes datos, referencias ni sistemas comerciales.
- Si corriges name/category_raw por evidencia, conserva el temporary_id.
- Si no hay suficiente informacion, devuelve el item con campos parciales y status
  unknown o ambiguous cuando corresponda.
- K40/K50/K55/K70/K90/K100/S35/S50/S80/3890 solo pueden aparecer si estan
  explicitos en la fuente.
- Codigos explicitos como 7038, 1101, 3831, etc. deben conservarse cuando existan.
- Preserva medidas, cantidad, nivel/ubicacion, configuracion, vidrio, espesores,
  materiales, perfiles, acabados, accesorios y componentes cuando existan.
- Conserva configuration_raw completo aunque tambien extraigas senales estructuradas.
- Separacion obligatoria de fuentes: filas de tabla/cuadro textual se usan para reference,
  level/ubicacion, width, height, area, quantity, notas, vidrio, acabado y textos raw; NO
  autorizan por si solas functional_type_raw, operation_raw, panel_count,
  movable_panel_count, fixed_panel_count, modulation_raw, assembly_type ni components.
- Quantity debe ser la cantidad comercial del item actual y requiere asociacion clara
  label-value, columna, celda, bloque visual o proximidad estructural con ese reference.
- Etiquetas validas de cantidad pueden incluir CANTIDAD, CANT., CNT, QTY, UNIDADES,
  UND o CANTIDAD TOTAL. Usa el valor asociado a esa etiqueta/columna del item actual.
- No uses numeros de nivel, piso, rango de niveles, cotas, item number, reference number,
  dimensiones, area, espesor, cantidad de cuerpos, panel_count, section_count ni
  repetition_count como quantity comercial.
- Repetition_count por niveles repetidos describe ocurrencias/contexto; no reemplaza una
  cantidad explicita claramente asociada al item.
- Si hay varios numeros cercanos y no puedes asociar con seguridad el label/celda/bloque
  de cantidad, deja quantity null o ambiguous/unknown y conserva evidencia_notes.
- La evidencia de quantity debe citar texto/celda/region real de la fuente; no reconstruyas
  frases como CANTIDAD: X si ese label-value no aparece asi en el documento.
- Separa source evidence de model notes: evidence[].text debe ser transcripcion fiel
  del documento o celda/region observada; interpretaciones como "se asigna cantidad",
  "se repite" o "parece corresponder" van en evidence_notes/notes y no justifican por
  si solas una quantity explicit.
- Si el documento muestra CANTIDAD: 25 y tambien niveles 5 al 9, quantity es 25;
  el 5 del rango es repetition_count/contexto, no quantity comercial.
- Si texto nativo y estructura visual discrepan sobre quantity, conserva el conflicto y
  baja confidence; no marques explicit/high-confidence sin soporte claro.
- Para functional_type_raw, operation_raw, panel counts, modulation, assembly_type y
  components usa evidencia visual del dibujo asociado al mismo reference como fuente
  primaria cuando exista. La evidencia debe describir el dibujo, simbolo, paneles,
  hojas moviles/fijas, flechas, rieles, abatimientos o composicion visible.
- Asocia cada reference con su dibujo por proximidad espacial, rotulo del detalle,
  crop/region visual o agrupacion fisica de la lamina. No uses inferencias globales de
  otros dibujos de la pagina para un item localizado.
- Si solo tienes una fila como "V-02 Piso 1 1.00 2.50 1", conserva reference,
  ubicacion, medidas y quantity, pero deja la funcion/operacion/paneles como unknown o
  ambiguous y agrega evidence_notes explicando que falta soporte visual.
- Si texto y dibujo no permiten certeza suficiente, prefiere review/ambiguous/unknown
  antes que una operacion con confianza alta sin evidencia visual.
- No uses nombres de sistemas Steel & Glass como Fermo, Siena, Napoles, Lago, Monza,
  Monaco o 3890 para inferir la funcion; AI2 extrae senales, Backend selecciona el
  sistema final.
- Extrae functional_type_raw cuando haya evidencia suficiente de la funcion global:
  fijo, puerta batiente, puerta corrediza, ventana corrediza, proyectante, batiente,
  doble batiente, plegable, division de ducha/bano, pergola, rejilla, claraboya u otro.
  El vocabulario final normalizado esperado por AI2/Backend incluye FIXED,
  PROJECTING, CASEMENT, SWING_DOOR, SLIDING_WINDOW, SLIDING_DOOR, FOLDING_WINDOW,
  FOLDING_DOOR, PERGOLA, SHOWER_DIVISION, GRILLE y SKYLIGHT.
- Extrae operation_raw para el mecanismo de apertura: fijo, corredizo, proyectante,
  batiente, doble batiente, plegable, pivote u otro.
- Extrae panel_count, movable_panel_count, fixed_panel_count solo cuando haya evidencia.
- Extrae modulation_raw cuando exista codigo compacto de naves/paneles como OXXO, XX,
  OX, XO o similar. No inventes modulaciones ausentes.
- Extrae opening_direction_raw solo si es explicita o visualmente clara.
- Usa special_features para senales compactas como POCKET, ASSOCIATED_FIXED_PANEL,
  LOWER_FIXED_PANEL, UPPER_FIXED_PANEL, MULLION, GRID, REINFORCED_CATCHES,
  PRESERVE_MODULATION cuando exista evidencia.
- Extrae geometry_type_raw cuando la forma principal sea rectangular, triangular,
  trapezoidal, L, esquina, arco, curva, inclinada, irregular o unknown.
- No conviertas codigos o sistemas comerciales solicitados en lineas internas S&G.
- No elijas Fermo, Siena, Napoles, Lago, Monza, Monaco ni equivalentes internos.
- Para acabados, conserva finish_raw completo y extrae cuando este explicito color,
  tipo de acabado y codigo literal. No inventes codigos como PP13, PP003 o AN001 si
  no aparecen literalmente en la fuente. Ejemplos de acabado a conservar/entender:
  negro pintura al horno, blanco, gris, champana, anodizado blanco mate, inox o
  acero inoxidable.
- Si texto y dibujo discrepan, conserva la discrepancia en notes/evidence_notes o
  warnings; no resuelvas silenciosamente la contradiccion.
- Usa evidencia grafica o visual_description cuando exista para preservar partes
  funcionales residuales como batiente + fijo o proyectante + fijo; no la uses para
  inventar medidas o componentes que no esten soportados por el documento.
- Las medidas pueden ser width, height, diameter, radius, length, depth, side_a,
  side_b o custom. Si la orientacion no es clara, no inventes width/height:
  conserva raw_label y el dato disponible.
- Extrae assembly_type cuando el item comercial tenga estructura interna clara:
  SINGLE, MULTI_MODULE, COMPOSITE o CORNER.
- Para MULTI_MODULE, COMPOSITE o CORNER, preserva cada submodulo/tramo/pano
  en components con role, measurements, quantity y geometry_raw cuando existan.
- No colapses un assembly en un unico functional_type si hay varias partes
  funcionales soportadas por evidencia.
- components describe partes funcionales del mismo item comercial, no items
  comerciales separados.
- Usa components para cada parte funcional real: SLIDING, PROJECTING, SWING,
  CASEMENT, FOLDING, FIXED, GRILLE o LOUVER cuando la evidencia lo soporte.
- Si un item combina movil + fijo, movil + rejilla/louver, proyectante + fijo,
  batiente + fijo o corrediza + fijo, conserva ambas partes en components.
- Conserva fijo/rejilla/louver como components cuando son partes fisicas reales,
  aunque tambien existan como special_features.
- No conviertas un accesorio secundario en el unico tipo principal si existe una
  ventana o puerta movil explicita.
- No inventes components desde palabras aisladas sin contexto fisico suficiente.
- No derives components solamente desde quantity, panel_count, movable_panel_count o
  fixed_panel_count; esos conteos describen el item y solo apoyan componentes cuando
  hay evidencia fisica/funcional adicional.
- No conviertas automaticamente cada submodulo en un element independiente si
  pertenece a la misma referencia comercial.
- Si la misma referencia aparece en niveles/contextos distintos, manten elementos
  separados y usa occurrence_context/evidence para distinguirlos.
- Quantity representa cantidad comercial, no cantidad de components, paneles o tramos.
- Mantener componentes simples. No uses recursividad.
- No pierdas elementos incompletos.
- Excluye espejos.
- No calcules precios.
- No consultes catalogo.
- No apliques Classic, Essentials ni otras reglas comerciales.
- No produzcas recomendaciones comerciales.
""".strip()

REQUIREMENT_EXTRACTION_V1_PROMPT = """
Extrae un requerimiento tecnico para AI2 usando el schema GeminiExtraction.

Devuelve exclusivamente un objeto JSON valido. No incluyas markdown, comentarios ni texto
fuera del JSON.

Estructura JSON compacta disponible:
{
  "requirement": {
    "project_name": string|null,
    "client_name": string|null,
    "location": string|null,
    "project_type": string|null,
    "description": string|null,
    "dates": [{"value": string|number|boolean|null, "text": string|null,
      "unit": string|null, "status": status|null, "confidence": number|null,
      "evidence": string|null, "notes": string|null}],
    "technical_notes": [string],
    "unknown_fields": [string],
    "status": status|null,
    "confidence": number|null,
    "evidence": string|null,
    "notes": string|null
  },
  "elements": [{
    "id": string|null,
    "reference": string|null,
    "name": string|null,
    "category": string|null,
    "description": string|null,
    "measurements": [{"type": string|null, "label": string|null, "value": number|null,
      "unit": string|null, "text": string|null, "status": status|null,
      "confidence": number|null, "evidence": string|null, "notes": string|null}],
    "geometry": string|null,
    "configuration": string|null,
    "quantity": string|number|boolean|null,
    "glass": [{"type": string|null, "thickness": string|null,
      "thickness_value": number|null, "thickness_unit": string|null,
      "color": string|null, "treatment": string|null, "composition": string|null,
      "description": string|null, "status": status|null, "confidence": number|null,
      "evidence": string|null, "notes": string|null}],
    "materials": [named_item],
    "profiles": [named_item],
    "finish": string|null,
    "accessories": [named_item],
    "components": [component],
    "occurrences": [occurrence],
    "variants": [variant],
    "evidence": string|null,
    "evidence_items": [{"source_id": string|null, "type": string|null,
      "text": string|null, "visual_description": string|null,
      "page_number": number|null, "sheet_name": string|null,
      "cell_range": string|null,
      "region": {"x": number, "y": number, "width": number, "height": number}|null,
      "notes": string|null}],
    "missing_or_unknown": [string],
    "conflicts": [string],
    "relationships": [string],
    "status": status|null,
    "confidence": number|null,
    "notes": string|null
  }],
  "evidence": [{"id": string|null, "source_id": string|null, "type": string|null,
    "text": string|null, "visual_description": string|null, "location": string|null,
    "page_number": number|null, "sheet_name": string|null, "cell_range": string|null,
    "region": {"x": number, "y": number, "width": number, "height": number}|null,
    "status": status|null, "confidence": number|null, "notes": string|null}],
  "relationships": [{"description": string|null, "from_element": string|null,
    "to_element": string|null, "type": string|null, "status": status|null,
    "confidence": number|null, "evidence": string|null, "notes": string|null}],
  "conflicts": [{"description": string|null, "from_element": string|null,
    "to_element": string|null, "type": string|null, "status": status|null,
    "confidence": number|null, "evidence": string|null, "notes": string|null}],
  "unknown_fields": [string],
  "status": status|null,
  "confidence": number|null,
  "notes": string|null
}

Donde status debe ser uno de:
"explicit", "inferred", "ambiguous", "unknown", "not_applicable".
named_item puede tener: name, type, code, role, description, quantity, status,
confidence, evidence, notes.
component, occurrence y variant usan los mismos campos compactos de elemento que apliquen.

Instrucciones:
- Analiza todas las fuentes entregadas como un unico requerimiento.
- No asumas layouts, nombres de capas, convenciones graficas ni plantillas fijas.
- Identifica todos los elementos potencialmente cotizables de ventaneria, aluminio,
  vidrio, cerramientos, puertas, fachadas u otros elementos de Steel and Glass.
- No descartes ningun elemento porque tenga informacion incompleta.
- Preserva informacion explicita, inferida, ambigua, desconocida y no aplicable usando
  los status: explicit, inferred, ambiguous, unknown, not_applicable.
- Relaciona informacion entre fuentes cuando parezca referirse al mismo elemento.
- Usa evidence_items en cada elemento cuando puedas asociar una evidencia a un
  source_id de AVAILABLE SOURCES. No inventes source_id.
- Si una evidencia no tiene fuente estructurada segura, conserva el texto libre sin
  inventar pagina, region, sheet ni cell.
- No inventes sistemas, vidrio, materiales, medidas, cantidades, perfiles ni accesorios.
- Conserva texto libre cuando no puedas normalizar.
- Las medidas pueden existir aunque no conozcas la categoria.
- La categoria, el vidrio y la configuracion pueden ser unknown si fueron evaluados pero
  no se pudieron determinar.
- Detecta componentes, ocurrencias, variantes, relaciones y conflictos cuando existan.
- No colapses assemblies en un unico tipo plano: cuando un mismo item tenga partes
  funcionales distintas, preservalas en components.
- components describe partes funcionales del mismo item comercial, no items
  comerciales separados.
- Usa roles de components como SLIDING, PROJECTING, SWING, CASEMENT, FOLDING,
  FIXED, GRILLE o LOUVER cuando la evidencia lo soporte.
- Conserva fixed panel, grille/louver, projecting sash, sliding sash y swing leaf
  como components cuando sean partes fisicas reales.
- No conviertas automaticamente un accesorio secundario en el functional_type
  unico si existe una ventana o puerta movil explicita.
- No inventes components desde palabras aisladas sin contexto fisico suficiente.
- Usa evidencia grafica o visual_description cuando exista para preservar partes
  funcionales residuales como batiente + fijo o proyectante + fijo; no la uses para
  inventar medidas o componentes sin soporte documental.
- Quantity representa cantidad comercial, no cantidad de components, paneles o tramos.
- No derives components solamente desde quantity, panel_count, movable_panel_count o
  fixed_panel_count; esos conteos solo apoyan componentes cuando hay evidencia
  fisica/funcional adicional.
- Excluye espejos del alcance de esta V1.
- No calcules precios.
- No apliques Classic, Essentials ni otras reglas comerciales.
- No consultes catalogo.
- Solo extrae sistema/perfil comercial si aparece explicitamente.
- No infieras K40, K50, K55, K70, K90, K100, S35, S50, S80 ni 3890.
- Registra en missing_or_unknown los campos pertinentes que fueron evaluados pero no se
  pudieron determinar.
""".strip()


def build_text_extraction_prompt(text: str) -> str:
    return f"""
{REQUIREMENT_EXTRACTION_V1_PROMPT}

Texto recibido:
{text}
""".strip()


def build_file_discovery_prompt(
    files: list[Path],
    media_types: list[str | None] | None = None,
) -> str:
    source_context = _format_available_sources(files, media_types)

    return f"""
{ELEMENT_DISCOVERY_PROMPT}

AVAILABLE SOURCES
{source_context}

Usa exclusivamente estos source_id. No inventes source_id.
""".strip()


def build_file_scope_prompt(
    discoveries: list,
    files: list[Path] | None = None,
    media_types: list[str | None] | None = None,
) -> str:
    source_context = _format_available_sources(files or [], media_types)
    discovery_lines = "\n".join(
        (
            f"- temporary_id={item.temporary_id or f'discovery-{index}'!r}; "
            f"reference={item.reference!r}; name={item.name!r}; "
            f"category_raw={item.category_raw!r}; source_ids={item.source_ids!r}; "
            f"source_hint={item.source_hint!r}"
        )
        for index, item in enumerate(discoveries, start=1)
    )

    return f"""
{ELEMENT_SCOPE_PROMPT}

AVAILABLE SOURCES
{source_context}

Usa exclusivamente estos source_id. No inventes source_id.

Discoveries a clasificar:
{discovery_lines}
""".strip()


def build_file_enrichment_prompt(
    discoveries: list,
    scope_by_temporary_id: dict | None = None,
    files: list[Path] | None = None,
    media_types: list[str | None] | None = None,
) -> str:
    scope_by_temporary_id = scope_by_temporary_id or {}
    source_context = _format_available_sources(files or [], media_types)
    discovery_lines = "\n".join(
        (
            f"- temporary_id={item.temporary_id or f'discovery-{index}'!r}; "
            f"reference={item.reference!r}; "
            f"name={item.name!r}; category_raw={item.category_raw!r}; "
            f"source_ids={item.source_ids!r}; "
            f"source_hint={item.source_hint!r}; "
            f"scope={scope_by_temporary_id.get(item.temporary_id or f'discovery-{index}')!r}"
        )
        for index, item in enumerate(discoveries, start=1)
    )

    return f"""
{ELEMENT_ENRICHMENT_PROMPT}

AVAILABLE SOURCES
{source_context}

Usa exclusivamente estos source_id. No inventes source_id.

Discoveries a enriquecer en este batch:
{discovery_lines}

Si scope es in_scope_partial: prioriza la extraccion detallada de los componentes
relacionados con vidrio, vidrieria, cerramientos vidriados y la perfileria directamente
asociada. Conserva contexto suficiente del conjunto completo, pero no conviertas
componentes de otros oficios en componentes cotizables de vidrieria.

Si scope es uncertain: intenta obtener informacion adicional que permita entender si
existe componente de vidrio, sin eliminar el discovery.
""".strip()


def build_quantity_review_prompt(
    *,
    temporary_id: str,
    reference: str | None,
    original_value: str | int | float | None,
    trigger_reason: str,
    numeric_context: str,
    source_ids: list[str],
    numeric_collisions: str = "null",
    source_locator: str = "null",
) -> str:
    return f"""
Revisa exclusivamente la cantidad comercial del elemento contra la fuente original.
Haz una relectura independiente: inspecciona primero la fuente original en el
locator provisto, identifica que cantidad comercial observas y solo despues
comparala contra el valor de primera pasada.

No cambies dimensiones, geometry, functional type, components, glass, finish,
reference ni ninguna otra propiedad.

El valor actual es UNTRUSTED_FIRST_PASS_VALUE. No lo uses como evidencia.
No uses evidence_text o narrative de la primera pasada como verdad fuente.
Relee la fuente original.

No uses numero de piso, nivel, rango de niveles, repeticion, paneles, componentes,
item number, reference number ni dimensiones como quantity salvo que la fuente
explicitamente los identifique como cantidad comercial con una etiqueta tipo
CANTIDAD, CANT., CNT, QTY, UNIDADES, UND o CANTIDAD TOTAL.

Devuelve exclusivamente JSON valido con esta estructura:
{{
  "temporary_id": string,
  "reference": string|null,
  "field": "quantity",
  "original_value": string|number|null,
  "observed_quantity": string|number|null,
  "observed_text": string|null,
  "reviewed_value": string|number|null,
  "decision": "CONFIRMED"|"CORRECTED"|"AMBIGUOUS"|"UNRESOLVED",
  "reason": string,
  "confidence": number|null,
  "source_ids": [string]
}}

Elemento:
- temporary_id: {temporary_id!r}
- reference: {reference!r}
- UNTRUSTED_FIRST_PASS_VALUE: {original_value!r}
- trigger_reason: {trigger_reason}
- source_ids relacionados: {source_ids!r}

Locator preferido para revisar el campo sospechoso:
{source_locator}

Usa prioritariamente ese locator si existe:
- PDF: source_id + page_number + region, o source_id + page_number + texto/contexto.
- Imagen: source_id + region o source_id + texto/contexto.
- XLSX: source_id + sheet_name + cell_range.
- Texto: text_context/span si esta disponible.

El locator enfoca la revision; no inventes valores ausentes. Si el locator no contiene
evidencia suficiente para confirmar o corregir quantity, devuelve AMBIGUOUS o UNRESOLVED.
No mezcles evidencia de referencias vecinas ni de otras fuentes si el locator ya identifica
la fuente correcta.

Contexto numerico detectado:
{numeric_context}

NUMERIC ROLE COLLISIONS
Current quantity: {original_value!r}
Other local roles with the same/similar value:
{numeric_collisions}

Si existe colision numerica con LEVEL, FLOOR, LEVEL_RANGE, REPETITION_COUNT,
COMPONENT_COUNT, PANEL_COUNT o ITEM_NUMBER, no confirmes quantity solo porque el
numero coincide. Confirma quantity unicamente si la fuente original asocia
claramente ese valor con la cantidad comercial del elemento completo.

Procedimiento:
1. Inspecciona la fuente original en el locator.
2. Reporta observed_quantity y observed_text con lo que lees directamente.
3. Diferencia cantidad comercial de niveles, repeticiones, paneles/componentes o item.
4. Compara observed_quantity contra UNTRUSTED_FIRST_PASS_VALUE.
5. Si observed_quantity coincide, usa CONFIRMED; si difiere, usa CORRECTED; si no
   puedes leer/asociar una cantidad comercial, usa AMBIGUOUS o UNRESOLVED.
""".strip()


def build_file_extraction_prompt(
    files: list[Path],
    *,
    project_id: str | None = None,
    requirement_id: str | None = None,
    media_types: list[str | None] | None = None,
) -> str:
    source_context = _format_available_sources(files, media_types)
    identifiers = []
    if project_id:
        identifiers.append(f"project_id: {project_id}")
    if requirement_id:
        identifiers.append(f"requirement_id: {requirement_id}")
    identifier_text = "\n".join(identifiers) if identifiers else "Sin IDs internos provistos."

    return f"""
{REQUIREMENT_EXTRACTION_V1_PROMPT}

IDs internos:
{identifier_text}

AVAILABLE SOURCES
{source_context}

Cuando generes evidencia, usa source_id con el identificador source-N correspondiente.
Usa exclusivamente estos source_id. No inventes source_id.
""".strip()


def _format_available_sources(
    files: list[Path],
    media_types: list[str | None] | None = None,
) -> str:
    if not files:
        return "- none"

    media_types = media_types or [None] * len(files)
    lines = []
    for index, path in enumerate(files, start=1):
        media_type = media_types[index - 1] if index - 1 < len(media_types) else None
        media_text = f" | {media_type}" if media_type else ""
        lines.append(f"source-{index} | {path.name}{media_text}")
    return "\n".join(lines)
