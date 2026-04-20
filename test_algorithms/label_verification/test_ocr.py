import argparse
import json
import os
import shutil
from pathlib import Path

import cv2
import pytesseract


THIS_FILE = Path(__file__).resolve()
TEST_ALGORITHMS_DIR = THIS_FILE.parents[1]
PROJECT_ROOT = THIS_FILE.parents[2]

if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

import label_verifier.controller as label_controller_module

from label_verifier.controller import VerificationController


YOLO_REF_DIR = TEST_ALGORITHMS_DIR / "data" / "yolo_ref"
OCR_REF_DIR = TEST_ALGORITHMS_DIR / "data" / "ocr_ref"
OCR_IMAGES_DIR = OCR_REF_DIR / "images"
OCR_COMPARE_DIR = OCR_REF_DIR / "compare_outputs"
OCR_EVIDENCE_DIR = OCR_COMPARE_DIR / "evidence"
OCR_MANIFEST_PATH = OCR_REF_DIR / "ocr_manifest.json"
SOURCE_MANIFEST_PATH = YOLO_REF_DIR / "annotation_manifest.json"
CONFIG_PATH = PROJECT_ROOT / "configs" / "default_config.ini"
DEFAULT_TARGET_TEXT = "REF"
DEFAULT_ICON = PROJECT_ROOT / "Icon Library" / "5_1_6_Catalogue_number.png"
DEFAULT_TESSERACT_EXE = Path(os.environ.get("TESSERACT_EXE", r"C:\Users\320308180\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clear_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_token(text: str) -> str:
    return "".join(ch for ch in text.upper() if ch.isalnum())


def rotate_image(image, rotation: int):
    if rotation == 0:
        return image
    if rotation == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"Unsupported rotation: {rotation}")


def preprocess_for_ocr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def score_token(token: str, target: str) -> float:
    norm = normalize_token(token)
    if not norm:
        return 0.0
    return 1.0 if norm == target else 0.0


def run_tesseract(image, tesseract_exe: Path, psm: int, target: str):
    pytesseract.pytesseract.tesseract_cmd = str(tesseract_exe)
    whitelist = "".join(sorted(set(target + "[]()")))
    config = f"--oem 3 --psm {psm} -c tessedit_char_whitelist={whitelist}"
    return pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config=config)


def find_text_match(image_path: Path, target_text: str, tesseract_exe: Path) -> dict:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return {
            "found": False,
            "score": 0.0,
            "rotation": 0,
            "text": "",
            "boxes": [],
            "matched_boxes": [],
            "render_image": None,
        }

    target = normalize_token(target_text)
    best = {
        "found": False,
        "score": 0.0,
        "rotation": 0,
        "text": "",
        "boxes": [],
        "matched_boxes": [],
        "render_image": image,
    }

    for rotation in (0, 90, 180, 270):
        rotated = rotate_image(image, rotation)
        processed = preprocess_for_ocr(rotated)
        for psm in (11, 7, 6):
            data = run_tesseract(processed, tesseract_exe, psm, target)
            boxes = []
            matched_boxes = []
            best_local_score = 0.0
            best_local_text = ""
            for idx, text in enumerate(data.get("text", [])):
                clean = text.strip()
                if not clean:
                    continue
                conf_raw = data.get("conf", ["-1"])[idx]
                try:
                    conf = max(0.0, float(conf_raw)) / 100.0
                except Exception:
                    conf = 0.0
                x = int(data["left"][idx])
                y = int(data["top"][idx])
                w = int(data["width"][idx])
                h = int(data["height"][idx])
                box = (x, y, x + w, y + h, clean, conf)
                boxes.append(box)
                sim = score_token(clean, target)
                score = sim * conf
                if sim >= 1.0:
                    matched_boxes.append(box)
                if score > best_local_score:
                    best_local_score = score
                    best_local_text = clean

            if best_local_score > best["score"]:
                best = {
                    "found": best_local_score >= 0.60 and len(matched_boxes) > 0,
                    "score": best_local_score,
                    "rotation": rotation,
                    "text": best_local_text,
                    "boxes": boxes,
                    "matched_boxes": matched_boxes,
                    "render_image": rotated,
                }

    return best


def save_ocr_evidence(image_name: str, result: dict, baseline_found: bool) -> str | None:
    image = result.get("render_image")
    if image is None:
        return None

    ensure_dir(OCR_EVIDENCE_DIR)
    overlay = image.copy()
    for x1, y1, x2, y2, text, conf in result.get("boxes", []):
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 165, 255), 1)
        cv2.putText(overlay, f"{text} {conf:.2f}", (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1, cv2.LINE_AA)
    for x1, y1, x2, y2, text, conf in result.get("matched_boxes", []):
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(overlay, f"MATCH {text} {conf:.2f}", (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

    status_lines = [
        f"Baseline: {'PASS' if baseline_found else 'FAIL'}",
        f"OCR: {'PASS' if result['found'] else 'FAIL'} score={result['score']:.2f}",
        f"Rotation: {result['rotation']} best_text={result['text']}",
    ]
    for index, line in enumerate(status_lines):
        y = 24 + index * 22
        cv2.putText(overlay, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(overlay, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 1, cv2.LINE_AA)

    evidence_path = OCR_EVIDENCE_DIR / image_name
    cv2.imwrite(str(evidence_path), overlay)
    return str(evidence_path)


def command_prepare(args) -> None:
    source_manifest = load_json(SOURCE_MANIFEST_PATH)
    clear_dir(OCR_IMAGES_DIR)
    ensure_dir(OCR_REF_DIR)

    items = {}
    count = 0
    for image_name, item in sorted(source_manifest.get("items", {}).items()):
        source_image = Path(item["image_path"])
        if not source_image.exists() or source_image.suffix.lower() != ".png":
            continue
        if args.source_split != "all" and item.get("split") != args.source_split:
            continue
        copied_path = OCR_IMAGES_DIR / image_name
        shutil.copy2(source_image, copied_path)
        items[image_name] = {
            "image_name": image_name,
            "image_path": str(copied_path),
            "source_pdf": item["source_pdf"],
            "page_index": int(item["page_index"]),
            "source_folder": item.get("source_folder", "unknown"),
            "original_split": item.get("split", "unknown"),
            "expected": True,
        }
        count += 1

    save_json(OCR_MANIFEST_PATH, {
        "target_text": args.target_text,
        "source_split": args.source_split,
        "items": items,
    })
    print(f"Prepared OCR images under: {OCR_IMAGES_DIR}")
    print(f"Image count: {count}")
    print(f"Manifest saved to: {OCR_MANIFEST_PATH}")


def run_baseline_for_pdf(baseline_controller: VerificationController, pdf_path: str, icon_path: str):
    inner = baseline_controller._inner
    original_store_debug = inner.config.get("store_debug_images", False)
    original_debug_dir = inner.config.get("debug_image_dir")
    original_build_report = label_controller_module.reporter.build_report
    original_build_icon_report = label_controller_module.reporter.build_icon_report

    inner.config["store_debug_images"] = False
    inner.config["debug_image_dir"] = None
    label_controller_module.reporter.build_report = lambda *args, **kwargs: None
    label_controller_module.reporter.build_icon_report = lambda *args, **kwargs: None
    try:
        results, _ = inner.run(
            input_paths=[pdf_path],
            icon_paths=[icon_path],
            output_path=None,
        )
    finally:
        inner.config["store_debug_images"] = original_store_debug
        inner.config["debug_image_dir"] = original_debug_dir
        label_controller_module.reporter.build_report = original_build_report
        label_controller_module.reporter.build_icon_report = original_build_icon_report

    return {int(item.page_index): item for item in results}


def command_compare(args) -> None:
    manifest = load_json(OCR_MANIFEST_PATH)
    tesseract_exe = Path(args.tesseract_exe)
    if not tesseract_exe.exists():
        raise FileNotFoundError(f"Tesseract executable not found: {tesseract_exe}")

    ensure_dir(OCR_COMPARE_DIR)
    clear_dir(OCR_EVIDENCE_DIR)
    results_path = OCR_COMPARE_DIR / "compare_results.json"
    if results_path.exists():
        results_path.unlink()
    baseline_controller = VerificationController(config_path=str(CONFIG_PATH))
    baseline_cache = {}
    metrics = {
        "baseline": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
        "ocr": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
    }
    rows = []
    target_text = args.target_text or manifest.get("target_text", DEFAULT_TARGET_TEXT)
    icon_path = str(DEFAULT_ICON.resolve())

    for image_name, item in sorted(manifest.get("items", {}).items()):
        expected = bool(item.get("expected", True))
        image_path = Path(item["image_path"])
        source_pdf = item["source_pdf"]
        page_index = int(item["page_index"])

        if source_pdf not in baseline_cache:
            baseline_cache[source_pdf] = run_baseline_for_pdf(baseline_controller, source_pdf, icon_path)
        baseline_result = baseline_cache[source_pdf].get(page_index)
        baseline_found = bool(baseline_result and str(baseline_result.decision).lower() == "pass")

        ocr_result = find_text_match(image_path, target_text, tesseract_exe)
        evidence_path = save_ocr_evidence(image_name, ocr_result, baseline_found)

        for name, found in (("baseline", baseline_found), ("ocr", ocr_result["found"])):
            if expected and found:
                metrics[name]["tp"] += 1
            elif expected and not found:
                metrics[name]["fn"] += 1
            elif not expected and found:
                metrics[name]["fp"] += 1
            else:
                metrics[name]["tn"] += 1

        rows.append({
            "image": image_name,
            "expected": expected,
            "baseline": baseline_found,
            "ocr": ocr_result["found"],
            "ocr_score": ocr_result["score"],
            "ocr_rotation": ocr_result["rotation"],
            "ocr_text": ocr_result["text"],
            "matched_box_count": len(ocr_result.get("matched_boxes", [])),
            "evidence_path": evidence_path,
        })
        print(
            f"{image_name} | GT={expected} | Baseline={'PASS' if baseline_found else 'FAIL'} | "
            f"OCR={'PASS' if ocr_result['found'] else 'FAIL'} ({ocr_result['score']:.2f}) | "
            f"Rotation={ocr_result['rotation']} | Text={ocr_result['text']}"
        )

    save_json(OCR_COMPARE_DIR / "compare_results.json", {"rows": rows, "metrics": metrics, "target_text": target_text})
    print("\nSummary:")
    for name, stat in metrics.items():
        print(f"- {name}: {stat}")
    print(f"Saved compare results to: {OCR_COMPARE_DIR / 'compare_results.json'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OCR-only REF symbol experiment: baseline vs OCR.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Copy image-only data from yolo_ref into ocr_ref.")
    prepare.add_argument("--source-split", choices=["all", "train", "val", "test"], default="all")
    prepare.add_argument("--target-text", default=DEFAULT_TARGET_TEXT)
    prepare.set_defaults(func=command_prepare)

    compare = subparsers.add_parser("compare", help="Compare baseline vs OCR on the prepared OCR image set.")
    compare.add_argument("--target-text", default=DEFAULT_TARGET_TEXT)
    compare.add_argument("--tesseract-exe", default=str(DEFAULT_TESSERACT_EXE))
    compare.set_defaults(func=command_compare)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
