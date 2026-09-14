"""Evaluación offline inicial del corpus. No importa app ni llama a servicios.

Los checks son datos de prueba, no reglas del extractor. Un valor que coincide no
certifica evidencia, estado, asociación espacial ni exactitud del documento entero.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
import zipfile
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Any

BASE = Path(__file__).resolve().parent
UNIT_FACTORS = {"m": Decimal("1"), "cm": Decimal("0.01"), "mm": Decimal("0.001")}


class InputError(ValueError):
    """Entrada inválida: no debe confundirse con una extracción incorrecta."""


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InputError(f"No se pudo leer JSON {path}: {exc}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reference_key(value: str | None) -> str:
    value = (value or "").strip().upper()
    match = re.fullmatch(r"([A-Z]{1,4})[\s._-]*(\d{1,4})([A-Z]?)", value)
    if match:
        return f"{match[1]}-{int(match[2])}{match[3]}"
    return value


def number(value: object) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, (str, int, float, Decimal)):
        return None
    try:
        parsed = Decimal(str(value).strip().replace(",", "."))
    except InvalidOperation:
        return None
    return parsed if parsed.is_finite() else None


def fold(value: object) -> str:
    if not isinstance(value, str):
        return ""
    text = unicodedata.normalize("NFKD", value.strip().casefold())
    return " ".join("".join(c for c in text if not unicodedata.combining(c)).split())


def verify_archive(archive_path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Comprueba bytes y nombres sin descomprimir, escribir ni ejecutar archivos."""
    expected = {
        unicodedata.normalize("NFC", doc["archive_path"]): doc
        for doc in manifest["documents"]
    }
    found: dict[str, zipfile.ZipInfo] = {}
    problems: list[str] = []
    file_results: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            name = unicodedata.normalize("NFC", member.filename.replace("\\", "/"))
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or ":" in name:
                raise InputError(f"Ruta no segura en ZIP: {name}")
            if name in found:
                raise InputError(f"Ruta duplicada en ZIP: {name}")
            found[name] = member
        for name, doc in expected.items():
            member = found.get(name)
            state = "MATCH"
            actual_hash = None
            if member is None:
                state = "MISSING"
            elif member.file_size != doc["size_bytes"]:
                state = "SIZE_MISMATCH"
            else:
                digest = hashlib.sha256()
                total = 0
                with archive.open(member) as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        total += len(block)
                        if total > doc["size_bytes"]:
                            raise InputError(f"Tamaño inesperado al leer {name}")
                        digest.update(block)
                actual_hash = digest.hexdigest()
                if actual_hash != doc["sha256"]:
                    state = "HASH_MISMATCH"
            if state != "MATCH":
                problems.append(f"{doc['document_id']}: {state}")
            file_results.append({
                "document_id": doc["document_id"], "state": state,
                "expected_sha256": doc["sha256"], "actual_sha256": actual_hash,
            })
    extras = sorted(set(found) - set(expected))
    if extras:
        problems.append("UNEXPECTED_FILES")
    return {
        "status": "INPUTS_MATCH" if not problems else "INPUTS_DIFFER",
        "documents_expected": len(expected),
        "documents_matched": sum(row["state"] == "MATCH" for row in file_results),
        "projects_registered": len(manifest["projects"]),
        "extra_files": extras, "problems": problems, "files": file_results,
        "note": "Verifica el ZIP, no la exactitud de AI2 ni qué bytes recibió un run previo.",
    }


def _read_scalar(element: dict[str, Any], field: str, stage: str) -> object:
    names = {
        "operation": "operation_raw" if stage == "enrichment" else "operation",
        "geometry_type": "geometry_type_raw" if stage == "enrichment" else "geometry_type",
    }
    return element.get(names.get(field, field))


def _check(element: dict[str, Any], check: dict[str, Any], stage: str) -> dict[str, Any]:
    kind = check["kind"]
    if kind == "measurement":
        candidates = [
            m for m in element["measurements"]
            if fold(m.get("type")) == check["field"]
        ]
        observed = [
            {k: m.get(k) for k in ("value", "unit", "status", "confidence")}
            for m in candidates
        ]
        target = number(check["expected"])
        target_unit = UNIT_FACTORS.get(check["unit"])
        if target is None or target_unit is None:
            raise InputError("Check de medidas inválido")
        converted = []
        for candidate in candidates:
            value = number(candidate.get("value"))
            factor = UNIT_FACTORS.get(fold(candidate.get("unit")))
            converted.append(value * factor if value is not None and factor else None)
        tolerance = Decimal(str(check.get("tolerance_m", "0.000001")))
        if not candidates:
            state = "MISSING"
        elif any(v is None for v in converted):
            state = "UNCOMPARABLE"
        elif all(abs(v - target * target_unit) <= tolerance for v in converted):
            state = "MATCH"
        else:
            state = "DIFFERENT"
    elif kind in {"number", "text_one_of"}:
        observed = _read_scalar(element, check["field"], stage)
        if observed is None:
            state = "MISSING"
        elif kind == "number":
            actual, target = number(observed), number(check["expected"])
            if actual is None or target is None:
                state = "UNCOMPARABLE"
            else:
                state = "MATCH" if actual == target else "DIFFERENT"
        else:
            matches = {fold(v) for v in check["expected"]}
            state = "MATCH" if fold(observed) in matches else "DIFFERENT"
    else:
        raise InputError(f"Tipo de check no soportado: {kind}")
    return {**check, "observed": observed, "state": state}


def evaluate_snapshot(
    snapshot: dict[str, Any], specification: dict[str, Any],
    project_id: str, stage: str = "enrichment",
) -> dict[str, Any]:
    """Compara sólo valores anotados; nunca modifica el snapshot recibido."""
    if stage not in {"enrichment", "pre-mapper"}:
        raise InputError("Etapa soportada: enrichment o pre-mapper")
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("elements"), list):
        raise InputError("Se requiere un snapshot con lista elements")
    by_reference: dict[str, list[dict[str, Any]]] = {}
    for element in snapshot["elements"]:
        if not isinstance(element, dict) or not isinstance(element.get("measurements"), list):
            raise InputError("Usa 02-enrichment o 04-pre-mapper, no discovery ni RAW de Backend")
        if element.get("reference") is not None and not isinstance(element["reference"], str):
            raise InputError("El contrato final RequirementExtraction aún no está soportado aquí")
        if not all(isinstance(m, dict) for m in element["measurements"]):
            raise InputError("measurements contiene entradas inválidas")
        by_reference.setdefault(reference_key(element.get("reference")), []).append(element)
    project = next((p for p in specification["projects"] if p["project_id"] == project_id), None)
    if project is None:
        raise InputError(f"Proyecto no registrado: {project_id}")
    rows: list[dict[str, Any]] = []
    expected_refs = project.get("expected_references")
    if expected_refs is not None:
        expected = Counter(reference_key(ref) for ref in expected_refs)
        actual = Counter({key: len(values) for key, values in by_reference.items()})
        rows.append({
            "check_id": "inventory", "field": "references", "expected": dict(expected),
            "observed": dict(actual), "state": "MATCH" if actual == expected else "DIFFERENT",
        })
    for case in project["elements"]:
        elements = by_reference.get(reference_key(case["reference"]), [])
        for check in case["checks"]:
            if len(elements) != 1:
                row = {
                    **check, "state": "MISSING" if not elements else "IDENTITY_AMBIGUOUS",
                    "observed": {"matching_elements": len(elements)},
                }
            else:
                row = _check(elements[0], check, stage)
            rows.append({
                "reference": case["reference"], "document_id": case["document_id"],
                "acceptance_case_id": case["acceptance_case_id"], **row,
            })
    counts = Counter(row["state"] for row in rows)
    return {
        "project_id": project_id, "stage": stage,
        "status": "PARTIAL_BASELINE" if rows else "NOT_ANNOTATED",
        "corpus_approved": False,
        "annotation_complete": False,
        "checked_values": len(rows), "matching_values": counts["MATCH"],
        "nonmatching_values": len(rows) - counts["MATCH"],
        "counts": dict(counts), "checks": rows,
        "not_evaluated": [
            "Campos y referencias no anotados", "Literalidad y localización de evidencia",
            "Estados de incertidumbre", "Conflictos", "Compatibilidad con Backend",
        ],
        "source_binding": "DECLARED_PROJECT_NOT_VERIFIED_BY_RUN_HASH",
        "note": "MATCH sólo indica coincidencia del check, no aprobación del elemento o corpus.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    verify = sub.add_parser("verify-inputs", help="Verificar el ZIP original sin extraerlo")
    verify.add_argument("archive", type=Path)
    verify.add_argument("--out", type=Path)
    baseline = sub.add_parser("baseline", help="Reutilizar el snapshot adjunto; sin Gemini")
    baseline.add_argument("--out", type=Path, default=BASE / "results" / "baseline.json")
    evaluate = sub.add_parser("evaluate", help="Comparar un snapshot existente")
    evaluate.add_argument("--snapshot", type=Path, required=True)
    evaluate.add_argument("--project", required=True, help="ID del proyecto en manifest.json")
    evaluate.add_argument("--stage", choices=["enrichment", "pre-mapper"], default="enrichment")
    evaluate.add_argument("--out", type=Path, default=BASE / "results" / "evaluation.json")
    evaluate.add_argument("--strict", action="store_true", help="Salir con 1 si hay diferencias")
    args = parser.parse_args(argv)
    try:
        if args.command == "verify-inputs":
            report = verify_archive(args.archive, read_json(BASE / "manifest.json"))
            exit_code = 0 if report["status"] == "INPUTS_MATCH" else 1
        else:
            snapshot_path = (
                BASE / "local_only" / "baseline_proyecto_1" / "02-enrichment.json"
                if args.command == "baseline" else args.snapshot
            )
            project_id = "proyecto_1" if args.command == "baseline" else args.project
            stage = "enrichment" if args.command == "baseline" else args.stage
            report = evaluate_snapshot(
                read_json(snapshot_path),
                read_json(BASE / "expected_checks.json"),
                project_id,
                stage,
            )
            report["snapshot_sha256"] = sha256_file(snapshot_path)
            report["snapshot_file"] = snapshot_path.name
            report["network_calls"] = 0
            report["extractor_execution"] = "NOT_RUN"
            exit_code = (
                1 if getattr(args, "strict", False)
                and (report["nonmatching_values"] or report["status"] == "NOT_ANNOTATED") else 0
            )
        if args.out:
            input_path = args.archive if args.command == "verify-inputs" else snapshot_path
            protected_paths = [
                input_path.resolve(), BASE / "manifest.json", BASE / "expected_checks.json",
                BASE / "acceptance_matrix.json", Path(__file__).resolve(),
            ]
            if args.out.resolve() in {path.resolve() for path in protected_paths}:
                raise InputError("El reporte no puede sobrescribir entradas ni configuración")
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        print(f"STATUS={report['status']}")
        for key in (
            "documents_expected", "documents_matched", "projects_registered",
            "checked_values", "matching_values", "nonmatching_values",
        ):
            if key in report:
                print(f"{key.upper()}={report[key]}")
        if "checks" in report:
            print("CORPUS_APPROVED=NO | EXTRACTION_RUN=NO | NETWORK_CALLS=0")
            for row in report["checks"]:
                if row["state"] != "MATCH":
                    print(f"  {row.get('reference', '*')} {row['field']}: {row['state']}")
        if args.out:
            print(f"REPORT={args.out}")
        return exit_code
    except (InputError, OSError, zipfile.BadZipFile, RuntimeError, KeyError) as exc:
        print(f"INPUT_ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
