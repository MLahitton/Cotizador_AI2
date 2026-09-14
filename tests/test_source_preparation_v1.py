"""Tests of preparation only; generated documents are fixtures, not client data."""
from __future__ import annotations

import hashlib
import json
import math
import socket
import stat
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from app.models.source_preparation_v1 import SourcePreparationOptions
from app.services.source_preparation_v1 import (
    SourcePreparationError,
    document_identity,
    prepare_source,
)
from manual_tests.corpus_v1.prepare_sources import prepare_corpus
from manual_tests.corpus_v1.runner import InputError


def make_pdf(path: Path, *, rotate=0, crop=None, text="V-TEST 1.80 CANT 2", pages=1,
             text_origin=(100, 150), text_rotation=False) -> Path:
    """Tiny self-contained PDF generator: no extra test dependencies/fonts/files."""
    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(pages))
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>",
            f"<< /Type /Pages /Kids [{kids}] /Count {pages} >>".encode(),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    for index in range(pages):
        crop_text = f" /CropBox [{' '.join(map(str, crop))}]" if crop else ""
        objs.append((f"<< /Type /Page /Parent 2 0 R /MediaBox [20 30 320 430] "
                     f"/Rotate {rotate}{crop_text} /Resources << /Font << /F1 3 0 R >> >> "
                     f"/Contents {5 + 2 * index} 0 R >>").encode())
        tm = f"0 1 -1 0 {text_origin[0]} {text_origin[1]} Tm" if text_rotation else \
             f"1 0 0 1 {text_origin[0]} {text_origin[1]} Tm"
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = (f"BT /F1 20 Tf 0 0 0 rg {tm} ({escaped}) Tj ET\n"
                  "0.7 0.7 0.7 RG 70 90 130 200 re S\n").encode()
        objs.append(f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"endstream")
    out = bytearray(b"%PDF-1.7\n")
    offsets = [0]
    for i, obj in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets[1:]:
        out += f"{offset:010} 00000 n \n".encode()
    out += (f"trailer\n<< /Root 1 0 R /Size {len(objs) + 1} >>\n"
            f"startxref\n{xref}\n%%EOF\n").encode()
    path.write_bytes(out)
    return path


def artifacts(root, result, view=0):
    directory = root / result.document_id
    return json.loads((directory / result.pages[view].observations_file).read_text("utf-8"))


def test_pdf_native_text_is_observation_and_source_unchanged(tmp_path):
    source = make_pdf(tmp_path / "case.pdf")
    before = source.read_bytes()
    result = prepare_source(source, tmp_path / "out")
    data = artifacts(tmp_path / "out", result)
    assert result.status == "PREPARED"
    assert result.content_sha256 == hashlib.sha256(before).hexdigest()
    assert source.read_bytes() == before
    assert "1.80" in data["raw_text_engine_order"]
    assert "CANT 2" in data["raw_text_engine_order"]
    assert data["sheet_number_from_drawing"] is None
    assert "quantity" not in data and "measurements" not in data
    assert all(char["visible_region"] is not None for char in data["characters"]
               if char["text"].isalnum() and not char["generated_by_text_engine"])
    assert result.provenance["network_calls"] == 0
    assert result.provenance["semantic_extraction_run"] is False


@pytest.mark.parametrize("rotate", [0, 90, 180, 270])
@pytest.mark.parametrize("crop", [None, (50, 70, 290, 380)])
def test_character_boxes_align_with_preview_for_pdf_rotation_and_crop(tmp_path, rotate, crop):
    source = make_pdf(tmp_path / "case.pdf", rotate=rotate, crop=crop, text="V")
    result = prepare_source(source, tmp_path / "out")
    data = artifacts(tmp_path / "out", result)
    assert data["frame"]["pdf_intrinsic_rotation_clockwise"] == rotate
    char = next(c for c in data["characters"] if c["text"] == "V")
    left, bottom, right, top = crop or (20, 30, 320, 430)
    x0, y0, x1, y1 = char["bbox_pdf_canvas"]
    x, y = (x0 + x1) / 2, (y0 + y1) / 2
    if rotate == 0:
        expected = ((x - left) / (right - left), (top - y) / (top - bottom))
    elif rotate == 90:
        expected = ((y - bottom) / (top - bottom), (x - left) / (right - left))
    elif rotate == 180:
        expected = ((right - x) / (right - left), (y - bottom) / (top - bottom))
    else:
        expected = ((top - y) / (top - bottom), (right - x) / (right - left))
    reg = char["visible_region"]
    assert reg["x"] + reg["width"]/2 == pytest.approx(expected[0], abs=0.003)
    assert reg["y"] + reg["height"]/2 == pytest.approx(expected[1], abs=0.003)
    with Image.open(tmp_path / "out" / result.document_id / result.pages[0].preview_file) as im:
        assert im.size == (data["frame"]["preview_width_px"], data["frame"]["preview_height_px"])
        # There really is non-white ink where the glyph box says it is.
        patch = im.crop(tuple(char["unclipped_bbox_px"])).convert("L")
        assert patch.getextrema()[0] < 128


def test_vertical_native_text_keeps_angle(tmp_path):
    result = prepare_source(make_pdf(tmp_path / "v.pdf", text="1.80", text_rotation=True),
                            tmp_path / "out")
    data = artifacts(tmp_path / "out", result)
    one = next(c for c in data["characters"] if c["text"] == "1")
    # Check the vertical orientation, without changing the raw engine angle convention.
    assert one["angle_radians_pdf_canvas"] % math.pi == pytest.approx(math.pi / 2, abs=0.01)


def test_outside_crop_text_not_silently_moved_onto_visible_page(tmp_path):
    result = prepare_source(make_pdf(tmp_path / "x.pdf", crop=(160, 200, 300, 400), text="V"),
                            tmp_path / "out")
    data = artifacts(tmp_path / "out", result)
    char = next(c for c in data["characters"] if c["text"] == "V")
    assert char["visible_region"] is None
    assert char["bbox_pdf_canvas"] is not None
    assert char["clipped_to_visible_page"] is True


def test_no_text_pdf_still_renders_and_requests_visual_reading(tmp_path):
    result = prepare_source(make_pdf(tmp_path / "scan.pdf", text=""), tmp_path / "out")
    assert result.status == "PREPARED"
    assert result.pages[0].native_char_count == 0
    assert result.pages[0].text_state == "NO_NATIVE_TEXT"
    assert "VISUAL_READING_REQUIRED_NO_OCR_RUN" in result.pages[0].warnings


def test_physical_pages_distinct_and_not_sheet_number(tmp_path):
    result = prepare_source(make_pdf(tmp_path / "D-37.pdf", pages=2), tmp_path / "out")
    assert result.page_count == 2
    assert [p.page_number for p in result.pages] == [1, 2]
    assert all(artifacts(tmp_path / "out", result, i)["sheet_number_from_drawing"] is None
               for i in (0, 1))


@pytest.mark.parametrize("orientation", range(1, 9))
def test_image_exif_orientation_and_no_fabricated_page(tmp_path, orientation):
    path = tmp_path / "photo.png"
    with Image.new("RGB", (80, 50), "white") as im:
        im.putpixel((0, 0), (0, 0, 0))
        exif = im.getexif()
        exif[274] = orientation
        im.save(path, exif=exif)
    before = path.read_bytes()
    result = prepare_source(path, tmp_path / "out")
    data = artifacts(tmp_path / "out", result)
    assert result.status == "PREPARED"
    assert result.page_count is None and result.pages[0].page_number is None
    expected = [50, 80] if orientation in (5, 6, 7, 8) else [80, 50]
    assert data["frame"]["oriented_size_px"] == expected
    assert data["frame"]["original_exif_orientation"] == orientation
    assert data["frame"]["visual_orientation_verified"] is False
    assert path.read_bytes() == before
    assert data["characters"] == []


def test_transparent_png_is_rendered_against_white(tmp_path):
    path = tmp_path / "alpha.png"
    with Image.new("RGBA", (20, 20), (0, 0, 0, 0)) as im:
        im.save(path)
    result = prepare_source(path, tmp_path / "out")
    with Image.open(tmp_path / "out" / result.document_id / "image.png") as im:
        assert im.getpixel((0, 0)) == (255, 255, 255)


def test_preview_budget_is_respected(tmp_path):
    options = SourcePreparationOptions(max_render_edge=128, max_render_pixels=10_000)
    result = prepare_source(make_pdf(tmp_path / "x.pdf"), tmp_path / "out", options=options)
    page = result.pages[0]
    assert max(page.width_px, page.height_px) <= 128
    assert page.width_px * page.height_px <= 10_000
    assert "PREVIEW_DOWNSCALED_TO_RESOURCE_LIMIT" in page.warnings


def test_character_limit_is_explicit_partial_not_silent_truncation(tmp_path):
    result = prepare_source(make_pdf(tmp_path / "x.pdf"), tmp_path / "out",
                            options=SourcePreparationOptions(max_chars_per_page=2))
    assert result.status == "PARTIAL"
    assert result.pages[0].text_state == "TEXT_EXTRACTION_FAILED"
    assert result.pages[0].preview_file is not None
    assert "NATIVE_TEXT_CHARACTER_LIMIT_EXCEEDED" in result.pages[0].warnings


def test_page_limit_does_not_silently_pass(tmp_path):
    result = prepare_source(make_pdf(tmp_path / "x.pdf", pages=2), tmp_path / "out",
                            options=SourcePreparationOptions(max_pages=1))
    assert result.status == "FAILED"
    assert result.page_count == 2
    assert "PDF_PAGE_LIMIT_EXCEEDED" in result.warnings


def test_corrupted_pdf_has_failed_source_artifact(tmp_path):
    source = tmp_path / "bad.pdf"
    source.write_bytes(b"%PDF-1.7\ninvalid file")
    result = prepare_source(source, tmp_path / "out")
    assert result.status == "FAILED"
    assert (tmp_path / "out" / result.document_id / "source.json").exists()


@pytest.mark.parametrize("data", [b"", b"NOT PDF", b"GIF89a"])
def test_bad_or_unsupported_source_rejected(tmp_path, data):
    path = tmp_path / "x.pdf"
    path.write_bytes(data)
    with pytest.raises(SourcePreparationError):
        prepare_source(path, tmp_path / "out")


def test_size_limit_and_hash_verification_before_write(tmp_path):
    source = make_pdf(tmp_path / "x.pdf")
    with pytest.raises(SourcePreparationError, match="SIZE_LIMIT"):
        prepare_source(source, tmp_path / "out", options=SourcePreparationOptions(max_file_bytes=5))
    with pytest.raises(SourcePreparationError, match="HASH"):
        prepare_source(source, tmp_path / "out", expected_sha256="0" * 64)
    assert not (tmp_path / "out").exists()


def test_output_collision_never_overwrites(tmp_path):
    path = make_pdf(tmp_path / "x.pdf")
    result = prepare_source(path, tmp_path / "out")
    target = tmp_path / "out" / result.document_id / "source.json"
    before = target.read_bytes()
    with pytest.raises(SourcePreparationError, match="OUTPUT_ALREADY_EXISTS"):
        prepare_source(path, tmp_path / "out")
    assert target.read_bytes() == before


def test_identity_independent_of_order_but_not_silently_deduplicated_by_bytes(tmp_path):
    path = make_pdf(tmp_path / "x.pdf")
    a = prepare_source(path, tmp_path / "one", logical_name="A/x.pdf", input_index=1)
    b = prepare_source(path, tmp_path / "two", logical_name="A/x.pdf", input_index=5)
    c = prepare_source(path, tmp_path / "three", logical_name="B/x.pdf", input_index=1)
    assert a.document_id == b.document_id
    assert a.document_id != c.document_id
    assert a.content_sha256 == c.content_sha256
    assert a.input_index != b.input_index


def test_unicode_identity_normalizes_path_only():
    assert document_identity("CARPINTERI\u0301A/a.pdf", "a" * 64) == \
           document_identity("CARPINTERÍA/a.pdf", "a" * 64)


@pytest.mark.parametrize("kwargs", [
    {"render_scale": 0}, {"render_scale": float("nan")}, {"render_scale": float("inf")},
    {"render_scale": 9}, {"max_pages": 0}, {"max_chars_per_page": False},
    {"max_render_edge": 9000}, {"max_render_pixels": 40_000_000},
])
def test_options_reject_bad_limits(kwargs):
    with pytest.raises(ValueError):
        SourcePreparationOptions(**kwargs)


def corpus_fixture(tmp_path):
    pdf = make_pdf(tmp_path / "a.pdf")
    png = tmp_path / "b.png"
    with Image.new("RGB", (10, 20), "white") as im:
        im.save(png)
    archive = tmp_path / "input.zip"
    docs = []
    with zipfile.ZipFile(archive, "w") as z:
        for i, path in enumerate((pdf, png), start=1):
            name = f"Requerimientos/project{i}/{path.name}"
            z.write(path, name)
            docs.append({"document_id": f"T{i}", "project_id": f"p{i}",
                         "archive_path": name, "size_bytes": path.stat().st_size,
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "pdf_pages": 1 if i == 1 else None})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"projects": [{"project_id": "p1"}, {"project_id": "p2"}],
                                    "documents": docs}), encoding="utf-8")
    return archive, manifest


def test_corpus_preparation_has_no_network_and_preserves_original(tmp_path, monkeypatch):
    archive, manifest = corpus_fixture(tmp_path)
    before = archive.read_bytes()
    def forbidden(*args, **kwargs):
        raise AssertionError("Network access is not allowed")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    report = prepare_corpus(archive, manifest, tmp_path / "out")
    assert report["documents_prepared"] == 2
    assert report["pdf_pages_rendered"] == 1
    assert report["image_views_rendered"] == 1
    assert report["semantic_extraction_run"] is False
    assert report["corpus_approved"] is False
    assert report["network_calls"] == 0
    assert archive.read_bytes() == before


def test_project_filter_and_unknown_project(tmp_path):
    archive, manifest = corpus_fixture(tmp_path)
    report = prepare_corpus(archive, manifest, tmp_path / "out", project="p1")
    assert report["documents_processed"] == 1
    assert report["projects_prepared"] == 1
    with pytest.raises(InputError, match="no registrado"):
        prepare_corpus(archive, manifest, tmp_path / "bad", project="other")


def test_integrity_mismatch_is_not_a_success(tmp_path):
    archive, manifest = corpus_fixture(tmp_path)
    data = json.loads(manifest.read_text())
    data["documents"][0]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(data))
    with pytest.raises(InputError, match="no coincide"):
        prepare_corpus(archive, manifest, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_zip_with_unsafe_path_rejected(tmp_path):
    archive, manifest = corpus_fixture(tmp_path)
    with zipfile.ZipFile(archive, "a") as z:
        z.writestr("../escape.pdf", b"x")
    with pytest.raises(InputError, match="no segura"):
        prepare_corpus(archive, manifest, tmp_path / "out")


def test_zip_symlink_rejected(tmp_path):
    archive, manifest = corpus_fixture(tmp_path)
    with zipfile.ZipFile(archive, "a") as z:
        info = zipfile.ZipInfo("link.pdf")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        z.writestr(info, "a.pdf")
    with pytest.raises(InputError, match="simbolicos"):
        prepare_corpus(archive, manifest, tmp_path / "out")
