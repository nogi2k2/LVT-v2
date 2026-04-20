import argparse
import json
import os
import random
import shutil
from pathlib import Path

import cv2
import yaml


THIS_FILE = Path(__file__).resolve()
TEST_ALGORITHMS_DIR = THIS_FILE.parents[1]
PROJECT_ROOT = THIS_FILE.parents[2]

if str(PROJECT_ROOT) not in os.sys.path:
	os.sys.path.insert(0, str(PROJECT_ROOT))

import label_verifier.controller as label_controller_module

from label_verifier.pdf_to_image import pdf_to_images
from label_verifier.controller import (
	VerificationController,
	Controller,
	_FILTERED_PAT_W,
	_FILTERED_SIG_W,
	_FILTERED_SIM_W,
	_cosine_sim,
	_safe_float,
)
from label_verifier import embeddings as emb_module
from label_verifier import pattern_verifier


LABELS_ROOT = PROJECT_ROOT / "data" / "Labels"
ICON_LIBRARY_ROOT = PROJECT_ROOT / "Icon Library"
CONFIG_PATH = PROJECT_ROOT / "configs" / "default_config.ini"
WORKSPACE_DIR = TEST_ALGORITHMS_DIR / "data" / "yolo_ref"
ANNOTATION_IMAGES_DIR = WORKSPACE_DIR / "annotation_images"
DATASET_DIR = WORKSPACE_DIR / "dataset_ref"
RUNS_DIR = WORKSPACE_DIR / "runs"
COMPARE_DIR = WORKSPACE_DIR / "compare_outputs"
COMPARE_EVIDENCE_DIR = COMPARE_DIR / "evidence"
MANIFEST_PATH = WORKSPACE_DIR / "annotation_manifest.json"
DATASET_YAML_PATH = DATASET_DIR / "dataset.yaml"
DEFAULT_TRAIN_SOURCE = "ECR 500044711_Labels"
DEFAULT_TEST_SOURCES = ["Model labels", "Package Labels"]
DEFAULT_ICON = ICON_LIBRARY_ROOT / "5_1_6_Catalogue_number.png"
DEFAULT_CLASS_NAME = "ref"
DEFAULT_RESHUFFLE_TRAIN_RATIO = 0.75
DEFAULT_RESHUFFLE_VAL_RATIO = 0.10
DEFAULT_RESHUFFLE_TEST_RATIO = 0.15


def sanitize(text: str) -> str:
	return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)


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


def write_classes_file(split_dir: Path, class_names: list[str]) -> None:
	ensure_dir(split_dir)
	(split_dir / "classes.txt").write_text("\n".join(class_names) + "\n", encoding="utf-8")


def load_manifest() -> dict:
	if not MANIFEST_PATH.exists():
		raise FileNotFoundError(f"Annotation manifest not found: {MANIFEST_PATH}")
	with open(MANIFEST_PATH, "r", encoding="utf-8") as handle:
		return json.load(handle)


def render_folder_to_images(folder_name: str, split: str, manifest: dict, limit: int | None = None) -> None:
	source_dir = LABELS_ROOT / folder_name
	if not source_dir.exists():
		raise FileNotFoundError(f"Label folder not found: {source_dir}")

	pdf_paths = sorted(source_dir.glob("*.pdf"))
	if limit is not None:
		pdf_paths = pdf_paths[:limit]

	target_dir = ANNOTATION_IMAGES_DIR / split
	ensure_dir(target_dir)

	for pdf_path in pdf_paths:
		page_images = pdf_to_images(str(pdf_path), dpi=300)
		for page_index, page_image in enumerate(page_images, start=1):
			image_name = f"{sanitize(folder_name)}__{sanitize(pdf_path.stem)}__page{page_index}.png"
			image_path = target_dir / image_name
			cv2.imwrite(str(image_path), page_image)
			manifest[image_name] = {
				"image_name": image_name,
				"image_path": str(image_path),
				"split": split,
				"source_folder": folder_name,
				"source_pdf": str(pdf_path),
				"page_index": page_index - 1,
			}


def command_prepare(args) -> None:
	random.seed(args.seed)
	clear_dir(ANNOTATION_IMAGES_DIR)
	ensure_dir(WORKSPACE_DIR)

	train_source_dir = LABELS_ROOT / args.train_source
	if not train_source_dir.exists():
		raise FileNotFoundError(f"Training folder not found: {train_source_dir}")

	train_pdfs = sorted(train_source_dir.glob("*.pdf"))
	if args.max_train_pdfs is not None:
		train_pdfs = train_pdfs[:args.max_train_pdfs]
	if not train_pdfs:
		raise ValueError(f"No PDFs found in {train_source_dir}")

	val_count = 0 if len(train_pdfs) == 1 else max(1, int(round(len(train_pdfs) * args.val_ratio)))
	val_set = set(pdf.name for pdf in train_pdfs[-val_count:]) if val_count else set()

	manifest = {
		"class_names": [args.class_name],
		"icon_path": str(Path(args.icon_path).resolve()),
		"train_source": args.train_source,
		"heldout_sources": args.heldout_sources,
		"items": {},
	}

	for pdf_path in train_pdfs:
		split = "val" if pdf_path.name in val_set else "train"
		page_images = pdf_to_images(str(pdf_path), dpi=300)
		target_dir = ANNOTATION_IMAGES_DIR / split
		ensure_dir(target_dir)
		for page_index, page_image in enumerate(page_images, start=1):
			image_name = f"{sanitize(args.train_source)}__{sanitize(pdf_path.stem)}__page{page_index}.png"
			image_path = target_dir / image_name
			cv2.imwrite(str(image_path), page_image)
			manifest["items"][image_name] = {
				"image_name": image_name,
				"image_path": str(image_path),
				"split": split,
				"source_folder": args.train_source,
				"source_pdf": str(pdf_path),
				"page_index": page_index - 1,
			}

	for heldout_source in args.heldout_sources:
		render_folder_to_images(heldout_source, "test", manifest["items"], limit=args.max_test_pdfs)

	save_json(MANIFEST_PATH, manifest)
	print(f"Prepared annotation images under: {ANNOTATION_IMAGES_DIR}")
	print(f"Manifest saved to: {MANIFEST_PATH}")
	print("Next:")
	print("1. Open labelImg")
	print(f"2. Open image dir: {ANNOTATION_IMAGES_DIR / 'train'}")
	print("3. Set save dir to the same folder")
	print("4. Use YOLO format and class name: ref")
	print("5. Draw a tight box around each REF symbol")
	print("6. Repeat for train, val, and test folders")


def copy_annotation_pair(image_path: Path, split: str) -> bool:
	label_path = image_path.with_suffix(".txt")
	if not label_path.exists():
		return False

	image_out = DATASET_DIR / "images" / split / image_path.name
	label_out = DATASET_DIR / "labels" / split / label_path.name
	ensure_dir(image_out.parent)
	ensure_dir(label_out.parent)
	shutil.copy2(image_path, image_out)
	shutil.copy2(label_path, label_out)
	return True


def command_build(args) -> None:
	manifest = load_manifest()
	clear_dir(DATASET_DIR)

	counts = {"train": 0, "val": 0, "test": 0}
	missing = []
	for split in counts:
		split_dir = ANNOTATION_IMAGES_DIR / split
		ensure_dir(split_dir)
		for image_path in sorted(split_dir.glob("*.png")):
			copied = copy_annotation_pair(image_path, split)
			if copied:
				counts[split] += 1
			else:
				missing.append(str(image_path))

	dataset_yaml = {
		"path": str(DATASET_DIR.resolve()),
		"train": "images/train",
		"val": "images/val",
		"test": "images/test",
		"names": manifest.get("class_names", [args.class_name]),
	}
	with open(DATASET_YAML_PATH, "w", encoding="utf-8") as handle:
		yaml.safe_dump(dataset_yaml, handle, sort_keys=False)

	print(f"YOLO dataset built at: {DATASET_DIR}")
	print(f"dataset.yaml: {DATASET_YAML_PATH}")
	print(f"Copied counts: {counts}")
	if missing:
		print(f"Images still missing annotations: {len(missing)}")
		for path in missing[:10]:
			print(f"- {path}")


def _read_annotation_stats(image_path: Path, label_path: Path) -> dict:
	image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
	if image is None:
		raise FileNotFoundError(f"Image not readable: {image_path}")

	height, width = image.shape[:2]
	areas = []
	for line in label_path.read_text(encoding="utf-8").splitlines():
		parts = line.strip().split()
		if len(parts) != 5:
			continue
		_, _x_center, _y_center, box_width, box_height = map(float, parts)
		areas.append(box_width * box_height)

	mean_bgr = image.mean(axis=(0, 1))
	blueish = bool(mean_bgr[0] > mean_bgr[1] + 8 and mean_bgr[0] > mean_bgr[2] + 8)

	return {
		"rel_area": max(areas) if areas else 0.0,
		"orientation": "portrait" if height >= width else "landscape",
		"blueish": blueish,
	}


def _size_bucket(rel_area: float, q1: float, q2: float) -> str:
	if rel_area <= q1:
		return "small"
	if rel_area <= q2:
		return "medium"
	return "large"


def _allocate_split_counts(count: int, train_ratio: float, val_ratio: float, test_ratio: float) -> dict:
	if count <= 1:
		return {"train": count, "val": 0, "test": 0}
	if count == 2:
		return {"train": 1, "val": 0, "test": 1}

	train = int(round(count * train_ratio))
	val = int(round(count * val_ratio))
	test = count - train - val

	if count >= 5:
		train = max(train, 2)
		val = max(val, 1)
		test = max(test, 1)
	else:
		train = max(train, 2)

	while train + val + test > count:
		largest = max(("train", train), ("val", val), ("test", test), key=lambda pair: pair[1])[0]
		if largest == "train" and train > 2:
			train -= 1
		elif largest == "val" and val > 1:
			val -= 1
		elif largest == "test" and test > 1:
			test -= 1
		else:
			train -= 1

	while train + val + test < count:
		if train <= val and train <= test:
			train += 1
		elif val <= test:
			val += 1
		else:
			test += 1

	return {"train": train, "val": val, "test": test}


def command_reshuffle(args) -> None:
	manifest = load_manifest()
	class_names = manifest.get("class_names", [DEFAULT_CLASS_NAME])
	random.seed(args.seed)

	entries = []
	for split in ("train", "val", "test"):
		split_dir = ANNOTATION_IMAGES_DIR / split
		ensure_dir(split_dir)
		for image_path in sorted(split_dir.glob("*.png")):
			label_path = image_path.with_suffix(".txt")
			if not label_path.exists():
				continue
			item = manifest["items"].get(image_path.name)
			if item is None:
				continue
			stats = _read_annotation_stats(image_path, label_path)
			entries.append({
				"name": image_path.name,
				"item": item,
				"image_path": image_path,
				"label_path": label_path,
				"source_folder": item.get("source_folder", "unknown"),
				"rel_area": stats["rel_area"],
				"orientation": stats["orientation"],
				"blueish": stats["blueish"],
			})

	if not entries:
		raise FileNotFoundError("No annotated image/label pairs found to reshuffle.")

	rel_areas = sorted(entry["rel_area"] for entry in entries)
	q1 = rel_areas[len(rel_areas) // 3]
	q2 = rel_areas[(2 * len(rel_areas)) // 3]

	for entry in entries:
		entry["size_bucket"] = _size_bucket(entry["rel_area"], q1, q2)
		entry["stratum"] = (
			entry["source_folder"],
			entry["size_bucket"],
			entry["orientation"],
			"blueish" if entry["blueish"] else "neutral",
		)

	staging_dir = WORKSPACE_DIR / "_reshuffle_staging"
	clear_dir(staging_dir)
	for entry in entries:
		staged_image = staging_dir / entry["image_path"].name
		staged_label = staging_dir / entry["label_path"].name
		shutil.move(str(entry["image_path"]), str(staged_image))
		shutil.move(str(entry["label_path"]), str(staged_label))
		entry["image_path"] = staged_image
		entry["label_path"] = staged_label

	for split in ("train", "val", "test"):
		clear_dir(ANNOTATION_IMAGES_DIR / split)
		write_classes_file(ANNOTATION_IMAGES_DIR / split, class_names)

	strata = {}
	for entry in entries:
		strata.setdefault(entry["stratum"], []).append(entry)

	assigned_counts = {"train": 0, "val": 0, "test": 0}
	for group_entries in strata.values():
		random.shuffle(group_entries)
		allocation = _allocate_split_counts(len(group_entries), args.train_ratio, args.val_ratio, args.test_ratio)
		index = 0
		for split in ("train", "val", "test"):
			for entry in group_entries[index:index + allocation[split]]:
				dest_dir = ANNOTATION_IMAGES_DIR / split
				dest_image = dest_dir / entry["image_path"].name
				dest_label = dest_dir / entry["label_path"].name
				shutil.move(str(entry["image_path"]), str(dest_image))
				shutil.move(str(entry["label_path"]), str(dest_label))
				entry["item"]["split"] = split
				entry["item"]["image_path"] = str(dest_image)
				assigned_counts[split] += 1
			index += allocation[split]

	if staging_dir.exists():
		shutil.rmtree(staging_dir)

	save_json(MANIFEST_PATH, manifest)
	print(f"Reshuffled annotated samples under: {ANNOTATION_IMAGES_DIR}")
	print(f"New split counts: {assigned_counts}")
	print(f"Size thresholds used: q1={q1:.6f}, q2={q2:.6f}")


def command_train(args) -> None:
	from ultralytics import YOLO

	if not DATASET_YAML_PATH.exists():
		raise FileNotFoundError(f"Build the dataset first: {DATASET_YAML_PATH}")

	ensure_dir(RUNS_DIR)
	model = YOLO(args.model)
	model.train(
		data=str(DATASET_YAML_PATH),
		epochs=args.epochs,
		imgsz=args.imgsz,
		batch=args.batch,
		project=str(RUNS_DIR),
		name=args.run_name,
		device=args.device,
		workers=args.workers,
		patience=args.patience,
		degrees=args.degrees,
		fliplr=args.fliplr,
		flipud=args.flipud,
		scale=args.scale,
		translate=args.translate,
		mosaic=args.mosaic,
	)
	print("Training finished.")


def load_test_ground_truth() -> dict:
	ground_truth = {}
	test_labels_dir = DATASET_DIR / "labels" / "test"
	if not test_labels_dir.exists():
		return ground_truth
	for label_path in sorted(test_labels_dir.glob("*.txt")):
		lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
		ground_truth[label_path.stem + ".png"] = len(lines) > 0
	return ground_truth


def load_gt_boxes(image_path: Path, image_shape) -> list[tuple[int, int, int, int]]:
	label_path = image_path.with_suffix(".txt")
	if not label_path.exists() or image_shape is None:
		return []

	height, width = image_shape[:2]
	boxes = []
	for line in label_path.read_text(encoding="utf-8").splitlines():
		parts = line.strip().split()
		if len(parts) != 5:
			continue
		_, x_center, y_center, box_width, box_height = map(float, parts)
		x1 = max(0, int(round((x_center - box_width / 2.0) * width)))
		y1 = max(0, int(round((y_center - box_height / 2.0) * height)))
		x2 = min(width - 1, int(round((x_center + box_width / 2.0) * width)))
		y2 = min(height - 1, int(round((y_center + box_height / 2.0) * height)))
		boxes.append((x1, y1, x2, y2))
	return boxes


def save_compare_evidence(
	image_path: Path,
	image,
	gt_boxes: list[tuple[int, int, int, int]],
	candidates: list[tuple[tuple[int, int, int, int], float]],
	best_box,
	best_conf: float,
	baseline_found: bool,
	yolo_found: bool,
	verify_found: bool,
	verify_score: float,
) -> str | None:
	if image is None:
		return None

	ensure_dir(COMPARE_EVIDENCE_DIR)
	overlay = image.copy()

	for x1, y1, x2, y2 in gt_boxes:
		cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 200, 0), 2)

	for box, conf in candidates:
		x1, y1, x2, y2 = [int(v) for v in box]
		cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 165, 255), 1)
		cv2.putText(
			overlay,
			f"cand {float(conf):.2f}",
			(x1, max(15, y1 - 6)),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.45,
			(0, 165, 255),
			1,
			cv2.LINE_AA,
		)

	if best_box is not None:
		x1, y1, x2, y2 = [int(v) for v in best_box]
		cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
		cv2.putText(
			overlay,
			f"best {best_conf:.2f}",
			(x1, max(15, y1 - 6)),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.5,
			(0, 0, 255),
			1,
			cv2.LINE_AA,
		)

	status_lines = [
		f"Baseline: {'PASS' if baseline_found else 'FAIL'}",
		f"YOLO: {'PASS' if yolo_found else 'FAIL'} ({best_conf:.2f})",
		f"YOLO+Verify: {'PASS' if verify_found else 'FAIL'} ({verify_score:.2f})",
	]
	for index, line in enumerate(status_lines):
		y = 24 + index * 22
		cv2.putText(overlay, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 3, cv2.LINE_AA)
		cv2.putText(overlay, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 1, cv2.LINE_AA)

	evidence_path = COMPARE_EVIDENCE_DIR / image_path.name
	cv2.imwrite(str(evidence_path), overlay)
	return str(evidence_path)


def load_crop_verifier(config_path: Path, icon_path: Path):
	verifier = VerificationController(config_path=str(config_path))._inner
	verifier._siglip_encoder = verifier._load_siglip()
	embedder_model = verifier._load_embedder(emb_module)
	icon_image = verifier._read_image(str(icon_path))
	if icon_image is None:
		raise FileNotFoundError(f"Reference icon not readable: {icon_path}")
	icon_emb = verifier._embed(emb_module, icon_image, embedder_model)
	icon_sig = verifier._siglip_embed(icon_image)
	return verifier, embedder_model, icon_image, icon_emb, icon_sig


def verify_crop(verifier: Controller, embedder_model, icon_image, icon_emb, icon_sig, crop):
	crop_emb = verifier._embed(emb_module, crop, embedder_model)
	crop_sig = verifier._siglip_embed(crop)
	sim_val = _cosine_sim(crop_emb, icon_emb) if crop_emb is not None and icon_emb is not None else None
	sig_val = verifier._siglip_similarity(crop_sig, icon_sig) if crop_sig is not None else None
	pat_score, _ = pattern_verifier.compute_combined_score(crop, icon_image)
	combined = (
		_FILTERED_SIM_W * _safe_float(sim_val) +
		_FILTERED_SIG_W * _safe_float(sig_val) +
		_FILTERED_PAT_W * _safe_float(pat_score)
	)
	sig_threshold = float(verifier.config.get("siglip_decision_threshold", 0.81))
	sim_threshold = float(verifier.config.get("sim_decision_threshold", 0.81))
	if sig_val is not None and verifier._siglip_encoder is not None:
		decision = _safe_float(sig_val) >= sig_threshold
		primary = _safe_float(sig_val)
	else:
		decision = sim_val is not None and _safe_float(sim_val) >= sim_threshold
		primary = _safe_float(sim_val)
	return {
		"found": bool(decision),
		"score": primary,
		"sim": _safe_float(sim_val),
		"siglip": _safe_float(sig_val),
		"combined": _safe_float(combined),
	}


def run_baseline_for_pdf(baseline_controller: VerificationController, pdf_path: str, icon_path: str, report_dir: Path):
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
	from ultralytics import YOLO

	manifest = load_manifest()
	ground_truth = load_test_ground_truth()
	if not ground_truth:
		raise FileNotFoundError("No test labels found. Annotate test images and run the build step first.")

	ensure_dir(COMPARE_DIR)
	baseline_controller = VerificationController(config_path=str(CONFIG_PATH))
	baseline_cache = {}
	icon_path = Path(args.icon_path).resolve()
	yolo_model = YOLO(args.weights)
	verifier_bundle = load_crop_verifier(CONFIG_PATH, icon_path) if args.verify_detections else None

	metrics = {
		"baseline": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
		"yolo": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
		"yolo_verified": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
	}
	rows = []

	for image_name, expected in sorted(ground_truth.items()):
		item = manifest["items"].get(image_name)
		if not item:
			continue

		image_path = Path(item["image_path"])
		source_pdf = item["source_pdf"]
		page_index = int(item["page_index"])
		if source_pdf not in baseline_cache:
			baseline_cache[source_pdf] = run_baseline_for_pdf(
				baseline_controller,
				source_pdf,
				str(icon_path),
				COMPARE_DIR / "baseline_reports",
			)
		baseline_result = baseline_cache[source_pdf].get(page_index)
		baseline_found = bool(baseline_result and str(baseline_result.decision).lower() == "pass")

		prediction = yolo_model.predict(source=str(image_path), imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
		image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
		gt_boxes = load_gt_boxes(image_path, image.shape if image is not None else None)
		yolo_found = False
		verify_found = False
		verify_score = 0.0
		best_conf = 0.0
		best_box = None
		candidates = []

		if prediction.boxes is not None and len(prediction.boxes) > 0:
			class_ids = prediction.boxes.cls.cpu().numpy().astype(int)
			confidences = prediction.boxes.conf.cpu().numpy()
			boxes = prediction.boxes.xyxy.cpu().numpy().astype(int)
			candidates = [
				(box, conf)
				for box, conf, class_id in zip(boxes, confidences, class_ids)
				if class_id == 0
			]
			if candidates:
				best_box, best_conf = max(candidates, key=lambda pair: float(pair[1]))
				yolo_found = True
				if verifier_bundle is not None and image is not None:
					x1, y1, x2, y2 = [int(v) for v in best_box]
					crop = image[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
					if crop.size:
						verify_result = verify_crop(*verifier_bundle, crop)
						verify_found = verify_result["found"]
						verify_score = verify_result["score"]

		def update_metric(name: str, found: bool):
			if expected and found:
				metrics[name]["tp"] += 1
			elif expected and not found:
				metrics[name]["fn"] += 1
			elif not expected and found:
				metrics[name]["fp"] += 1
			else:
				metrics[name]["tn"] += 1

		update_metric("baseline", baseline_found)
		update_metric("yolo", yolo_found)
		update_metric("yolo_verified", verify_found if verifier_bundle is not None else yolo_found)

		evidence_path = save_compare_evidence(
			image_path,
			image,
			gt_boxes,
			candidates,
			best_box,
			best_conf,
			baseline_found,
			yolo_found,
			verify_found if verifier_bundle is not None else yolo_found,
			verify_score,
		)

		rows.append({
			"image": image_name,
			"expected": expected,
			"baseline": baseline_found,
			"baseline_score": getattr(baseline_result, "score", None) if baseline_result else None,
			"yolo": yolo_found,
			"yolo_conf": float(best_conf),
			"yolo_verified": verify_found if verifier_bundle is not None else yolo_found,
			"verify_score": verify_score,
			"candidate_count": len(candidates),
			"evidence_path": evidence_path,
		})
		print(
			f"{image_name} | GT={expected} | Baseline={'PASS' if baseline_found else 'FAIL'} | "
			f"YOLO={'PASS' if yolo_found else 'FAIL'} ({best_conf:.2f}) | "
			f"YOLO+Verify={'PASS' if (verify_found if verifier_bundle is not None else yolo_found) else 'FAIL'}"
		)

	save_json(COMPARE_DIR / "compare_results.json", {"rows": rows, "metrics": metrics})
	print("\nSummary:")
	for name, stat in metrics.items():
		print(f"- {name}: {stat}")
	print(f"Saved compare results to: {COMPARE_DIR / 'compare_results.json'}")


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="YOLO-based label verification experiment for the REF symbol.")
	subparsers = parser.add_subparsers(dest="command", required=True)

	prepare = subparsers.add_parser("prepare", help="Convert label PDFs to images for annotation.")
	prepare.add_argument("--train-source", default=DEFAULT_TRAIN_SOURCE)
	prepare.add_argument("--heldout-sources", nargs="+", default=DEFAULT_TEST_SOURCES)
	prepare.add_argument("--class-name", default=DEFAULT_CLASS_NAME)
	prepare.add_argument("--icon-path", default=str(DEFAULT_ICON))
	prepare.add_argument("--val-ratio", type=float, default=0.2)
	prepare.add_argument("--seed", type=int, default=42)
	prepare.add_argument("--max-train-pdfs", type=int)
	prepare.add_argument("--max-test-pdfs", type=int)
	prepare.set_defaults(func=command_prepare)

	build = subparsers.add_parser("build", help="Build the YOLO dataset from annotated images.")
	build.add_argument("--class-name", default=DEFAULT_CLASS_NAME)
	build.set_defaults(func=command_build)

	reshuffle = subparsers.add_parser("reshuffle", help="Reshuffle all annotated samples across train/val/test.")
	reshuffle.add_argument("--seed", type=int, default=42)
	reshuffle.add_argument("--train-ratio", type=float, default=DEFAULT_RESHUFFLE_TRAIN_RATIO)
	reshuffle.add_argument("--val-ratio", type=float, default=DEFAULT_RESHUFFLE_VAL_RATIO)
	reshuffle.add_argument("--test-ratio", type=float, default=DEFAULT_RESHUFFLE_TEST_RATIO)
	reshuffle.set_defaults(func=command_reshuffle)

	train = subparsers.add_parser("train", help="Train a YOLO model on the prepared dataset.")
	train.add_argument("--model", default="yolo11n.pt")
	train.add_argument("--epochs", type=int, default=60)
	train.add_argument("--imgsz", type=int, default=1280)
	train.add_argument("--batch", type=int, default=8)
	train.add_argument("--device", default="cpu")
	train.add_argument("--workers", type=int, default=2)
	train.add_argument("--patience", type=int, default=20)
	train.add_argument("--run-name", default="ref_symbol_yolo")
	train.add_argument("--degrees", type=float, default=180.0)
	train.add_argument("--fliplr", type=float, default=0.0)
	train.add_argument("--flipud", type=float, default=0.5)
	train.add_argument("--scale", type=float, default=0.1)
	train.add_argument("--translate", type=float, default=0.05)
	train.add_argument("--mosaic", type=float, default=0.0)
	train.set_defaults(func=command_train)

	compare = subparsers.add_parser("compare", help="Compare LVT baseline vs YOLO on the test split.")
	compare.add_argument("--weights", required=True)
	compare.add_argument("--imgsz", type=int, default=1280)
	compare.add_argument("--conf", type=float, default=0.25)
	compare.add_argument("--icon-path", default=str(DEFAULT_ICON))
	compare.add_argument("--verify-detections", action="store_true")
	compare.set_defaults(func=command_compare)

	return parser


def main() -> None:
	parser = build_parser()
	args = parser.parse_args()
	args.func(args)


if __name__ == "__main__":
	main()
