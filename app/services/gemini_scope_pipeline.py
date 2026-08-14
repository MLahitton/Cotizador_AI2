from app.models.gemini_discovery import GeminiDiscoveryResult, GeminiElementDiscovery
from app.models.gemini_scope import GeminiScopeClassification, GeminiScopeResult, ScopeStatus


def merge_scope_with_discovery(
    discovery: GeminiDiscoveryResult,
    scope_result: GeminiScopeResult,
) -> GeminiScopeResult:
    warnings = list(scope_result.warnings)
    scope_by_id: dict[str, GeminiScopeClassification] = {}

    for scope in scope_result.elements:
        if scope.temporary_id in scope_by_id:
            warnings.append(f"duplicate scope temporary_id {scope.temporary_id!r}")
            continue
        scope_by_id[scope.temporary_id] = scope

    ordered_scopes = []
    for index, discovered in enumerate(discovery.elements, start=1):
        temporary_id = discovery_temporary_id(discovered, index)
        scope = scope_by_id.get(temporary_id)
        if scope is None:
            warnings.append(f"missing scope for temporary_id {temporary_id!r}; using uncertain")
            scope = GeminiScopeClassification(
                temporary_id=temporary_id,
                scope=ScopeStatus.UNCERTAIN,
                reason="Scope classification missing; preserved as uncertain.",
            )
        ordered_scopes.append(scope)

    return GeminiScopeResult(elements=ordered_scopes, warnings=warnings)


def select_discoveries_for_enrichment(
    discovery: GeminiDiscoveryResult,
    scope_result: GeminiScopeResult,
) -> GeminiDiscoveryResult:
    included_scope_ids = {
        scope.temporary_id
        for scope in scope_result.elements
        if scope.scope
        in {
            ScopeStatus.IN_SCOPE_FULL,
            ScopeStatus.IN_SCOPE_PARTIAL,
            ScopeStatus.UNCERTAIN,
        }
    }
    return GeminiDiscoveryResult(
        elements=[
            discovered
            for index, discovered in enumerate(discovery.elements, start=1)
            if discovery_temporary_id(discovered, index) in included_scope_ids
        ],
        notes=list(discovery.notes),
    )


def scope_lookup(scope_result: GeminiScopeResult) -> dict[str, str]:
    return {scope.temporary_id: scope.scope.value for scope in scope_result.elements}


def discovery_temporary_id(discovery: GeminiElementDiscovery, index: int) -> str:
    return discovery.temporary_id or f"discovery-{index}"
