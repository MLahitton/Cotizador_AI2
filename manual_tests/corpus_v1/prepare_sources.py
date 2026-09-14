"""Prepare the registered corpus locally, without API key or application servers.

Uses the corpus manifest ONLY to verify and organize fixtures. No reference or
expected answer from that manifest is consulted by the generic preparation service.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import stat
import subprocess
import sys
import tempfile
import unicodedata
import uuid
import zipfile
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parent
REPO_ROOT = BASE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.models.source_preparation_v1 import SourcePreparationOptions  # noqa: E402
from app.services.source_preparation_v1 import (  # noqa: E402
    SourcePreparationError,
    prepare_source,
)
from manual_tests.corpus_v1.runner import InputError, read_json, verify_archive  # noqa: E402


def _write_report(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _code_provenance() -> dict[str, Any]:
    paths = (
        "app/models/source_preparation_v1.py",
        "app/services/source_preparation_v1.py",
        "manual_tests/corpus_v1/prepare_sources.py",
    )
    commit = None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
            capture_output=True, timeout=5, check=False,
        )
        if result.returncode == 0:
            commit = result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return {
        "git_head": commit,
        "actual_code_sha256": {name: _file_hash(REPO_ROOT / name) for name in paths},
        "note": "Hashes identify executed files, including uncommitted code; HEAD alone does not.",
    }


def _precheck_zip(path: Path, options: SourcePreparationOptions) -> None:
    with zipfile.ZipFile(path) as archive:
        items = archive.infolist()
        if len(items) > 1000 or sum(item.file_size for item in items) > 250 * 1024 * 1024:
            raise InputError("ZIP supera los limites de esta preparacion local.")
        for member in items:
            if member.file_size > options.max_file_bytes:
                raise InputError("Un archivo del ZIP supera el limite de tamano.")
            if member.flag_bits & 1:
                raise InputError("ZIP cifrado no admitido por este runner.")
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise InputError("No se admiten enlaces simbolicos en el ZIP.")


def prepare_corpus(
    archive_path: Path,
    manifest_path: Path,
    output: Path,
    *,
    project: str | None = None,
    options: SourcePreparationOptions | None = None,
) -> dict[str, Any]:
    options = options or SourcePreparationOptions()
    archive_path, manifest_path, output = map(Path, (archive_path, manifest_path, output))
    _precheck_zip(archive_path, options)
    archive_hash_before = _file_hash(archive_path)
    manifest = read_json(manifest_path)
    integrity = verify_archive(archive_path, manifest)
    if integrity["status"] != "INPUTS_MATCH":
        raise InputError("El ZIP no coincide con el corpus: " + "; ".join(integrity["problems"]))
    if project is not None and project not in {p["project_id"] for p in manifest["projects"]}:
        raise InputError(f"Proyecto no registrado: {project}")
    selected = [d for d in manifest["documents"] if project is None or d["project_id"] == project]
    if output.exists() or output.is_symlink():
        raise InputError(
            "La carpeta de salida ya existe. Usa una carpeta nueva; no se sobrescribe."
        )
    output.mkdir(parents=True, exist_ok=False)
    _write_report(output / "inputs_verification.json", integrity)
    rows = []
    with (
        zipfile.ZipFile(archive_path) as archive,
        tempfile.TemporaryDirectory(prefix="ai2-src-") as tmp,
    ):
        members = {
            unicodedata.normalize("NFC", m.filename.replace("\\", "/")): (i, m)
            for i, m in enumerate((m for m in archive.infolist() if not m.is_dir()), start=1)
        }
        for index, doc in enumerate(selected, start=1):
            input_index, member = members[unicodedata.normalize("NFC", doc["archive_path"])]
            # Never extract using the archive path: write a generated safe temporary name.
            local = Path(tmp) / f"input-{index:04d}{Path(member.filename).suffix.lower()}"
            row = {"corpus_document_id": doc["document_id"], "project_id": doc["project_id"],
                   "archive_path": member.filename, "archive_input_index": input_index}
            try:
                with archive.open(member) as inp, local.open("xb") as out:
                    copied = 0
                    while block := inp.read(1024 * 1024):
                        copied += len(block)
                        if copied > doc["size_bytes"] or copied > options.max_file_bytes:
                            raise InputError("Tamaño del miembro ZIP inesperado.")
                        out.write(block)
                prepared = prepare_source(
                    local, output / "documents", logical_name=member.filename,
                    input_index=input_index, expected_sha256=doc["sha256"], options=options,
                )
                row.update(asdict(prepared))
                row["artifact_directory"] = f"documents/{prepared.document_id}"
                expected_pages = doc.get("pdf_pages")
                if expected_pages is not None and prepared.page_count != expected_pages:
                    row["status"] = "FAILED"
                    row["warnings"].append("PHYSICAL_PAGE_COUNT_DIFFERS_FROM_MANIFEST")
            except (InputError, SourcePreparationError, OSError) as exc:
                row.update({"status": "FAILED", "pages": [], "error": str(exc)})
            finally:
                local.unlink(missing_ok=True)
            rows.append(row)
            print(f"[{index}/{len(selected)}] {doc['document_id']} {doc['project_id']}: "
                  f"{row['status']}", flush=True)
    changed = _file_hash(archive_path) != archive_hash_before
    if changed:
        raise InputError("El ZIP cambio durante la preparacion. No usar estos resultados.")
    pages = [page for row in rows for page in row.get("pages", [])]
    by_status = Counter(row["status"] for row in rows)
    all_prepared = by_status["PREPARED"] == len(selected)
    report = {
        "schema_version": 1,
        "status": "PREPARATION_COMPLETED" if all_prepared else "PREPARATION_HAS_ERRORS",
        "run_id": output.name,
        "created_utc": datetime.now(UTC).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.system(),
        "source_archive_sha256": archive_hash_before,
        "source_archive_unchanged": True,
        "manifest_sha256": _file_hash(manifest_path),
        "projects_prepared": len({r["project_id"] for r in rows}),
        "documents_processed": len(rows),
        "documents_prepared": by_status["PREPARED"],
        "documents_partial": by_status["PARTIAL"],
        "documents_failed": by_status["FAILED"],
        "pdf_pages_rendered": sum(p["page_number"] is not None and p["preview_file"] is not None
                                  for p in pages),
        "image_views_rendered": sum(p["page_number"] is None and p["preview_file"] is not None
                                    for p in pages),
        "pdf_pages_with_native_text": sum(p["text_state"] == "NATIVE_TEXT_PRESENT_UNVERIFIED"
                                          for p in pages),
        "pdf_pages_without_native_text": sum(p["text_state"] == "NO_NATIVE_TEXT" for p in pages),
        "native_characters": sum(p["native_char_count"] for p in pages),
        "located_characters": sum(p["located_char_count"] for p in pages),
        "text_reading_errors": sum(p["text_state"] == "TEXT_EXTRACTION_FAILED" for p in pages),
        "options": asdict(options),
        "code_provenance": _code_provenance(),
        "semantic_extraction_run": False, "ocr_run": False, "network_calls": 0,
        "corpus_approved": False,
        "documents": rows,
        "note": "PREPARED certifies artifact creation only; not text accuracy or AI extraction.",
    }
    _write_report(output / "source_preparation_report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Ruta al Requerimientos.zip original")
    parser.add_argument("--manifest", type=Path, default=BASE / "manifest.json")
    parser.add_argument("--project", help="Opcional: project_id del manifiesto")
    parser.add_argument("--out", type=Path, help="Carpeta NUEVA para este run")
    args = parser.parse_args(argv)
    output = args.out or BASE / "results" / "source_preparation" / uuid.uuid4().hex
    try:
        # Fail early without importing providers/settings or reading environment secrets.
        import pypdfium2  # noqa: F401
        report = prepare_corpus(args.archive, args.manifest, output, project=args.project)
        for key in ("status", "projects_prepared", "documents_processed", "documents_prepared",
                    "documents_partial", "documents_failed", "pdf_pages_rendered",
                    "image_views_rendered", "pdf_pages_with_native_text",
                    "pdf_pages_without_native_text", "text_reading_errors"):
            print(f"{key.upper()}={report[key]}")
        print("CORPUS_APPROVED=NO | SEMANTIC_EXTRACTION_RUN=NO | OCR_RUN=NO | NETWORK_CALLS=0")
        print(f"REPORT={(output / 'source_preparation_report.json').resolve()}")
        return 0 if report["status"] == "PREPARATION_COMPLETED" else 1
    except ImportError:
        print('DEPENDENCY_MISSING: ejecutar uv add "pypdfium2>=5.8,<6".', file=sys.stderr)
        return 2
    except (InputError, SourcePreparationError, OSError, ValueError, zipfile.BadZipFile) as exc:
        print(f"PREPARATION_ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
