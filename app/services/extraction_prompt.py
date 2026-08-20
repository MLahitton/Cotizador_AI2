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
  text, page_number, sheet_name, cell_range o visual_description segun aplique.
- Usa exclusivamente los source_id listados en AVAILABLE SOURCES; no inventes source_id.
- Cuando una afirmacion provenga de mas de una fuente, puede indicar varias evidencias.
- Mantén separadas afirmaciones contradictorias entre fuentes; no fusiones evidencia
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
- Extrae functional_type_raw cuando haya evidencia suficiente de la funcion global:
  fijo, puerta corrediza, ventana corrediza, proyectante, batiente, doble batiente,
  plegable, division de bano, baranda, pergola, rejilla, claraboya, fachada u otro.
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
- Si texto y dibujo discrepan, conserva la discrepancia en notes/evidence_notes o
  warnings; no resuelvas silenciosamente la contradiccion.
- Las medidas pueden ser width, height, diameter, radius, length, depth, side_a,
  side_b o custom. Si la orientacion no es clara, no inventes width/height:
  conserva raw_label y el dato disponible.
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
      "cell_range": string|null, "notes": string|null}],
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
