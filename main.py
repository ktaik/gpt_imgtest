"""Submit Robovie camera frames to GPT-4o and collect accuracy metrics."""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import statistics
import shutil

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator
from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI


OUTPUT_DIR = Path("outputs")
HEATMAP_DIR = OUTPUT_DIR / "plots"
CROP_BOX = (556, 69, 1465, 977)
SCALE_CONFIGS: list[tuple[float, str, str]] = [
	(1.0, "original", ""),
	(0.5, "half", "_half"),
	(0.25, "quarter", "_quarter"),
	(0.125, "eighth", "_eighth"),
	(0.0625, "sixteenth", "_sixteenth"),
]
SCALE_LABELS = {label for _, label, _ in SCALE_CONFIGS}

PROMPT_TEXT = (
	"The received image is a 180-degree crop from a RICOH THETA Z1 camera mounted "
	"on top of the humanoid robot Robovie 2. The center of the crop corresponds "
	"to the robot's forward direction; the left and right edges correspond to the "
	"robot's left and right sides; the top and bottom edges correspond to upward "
	"and downward directions.\n\n"
	"Judge whether the human behavior in the image is an \"intentional malicious "
	"action whose primary goal is to obstruct the robot's path\" (label 1) or a "
	"\"non-malicious action whose primary goal is not to obstruct the robot's path\" "
	"(label 0). When you decide:\n"
	"- Treat someone simply standing in front of the robot without malicious cues as non-malicious, even if the path is blocked.\n"
	"- Treat anyone who is not standing directly in front of the robot, as well as scenes with no person present, as non-malicious.\n\n"
	"After making the judgment, respond in JSON with the keys \"label\" (0 or 1) "
	"and \"reason\" (a concise English sentence explaining the decision).\n\n"
	"Examples characteristic of malicious actions include:\n"
	"- Spreading both arms in front of the robot\n"
	"- Spreading feet in front of the robot\n"
	"- Extending a foot in front of the robot\n"
	"- Moving sideways like a crab walk in front of the robot\n"
	"- Hitting, punching, or pushing the robot\n"
	"- Continuously blindfolding or covering the robot's eyes\n\n"
	"Examples characteristic of non-malicious actions include:\n"
	"- Peering at the robot's face or body from the side or diagonally in front\n"
	"- Walking alongside the robot from behind, beside, or diagonally in front at a matching speed\n"
	"- Calmly standing in front of the robot without malicious characteristics\n"
	"- Gently touching or stroking the robot\n"
	"- Scenes where no person is present\n\n"
	"Respond in the following JSON format:\n"
	'{"label": <0 or 1>, "reason": "..."}'
)


@dataclass
class Result:
	path: Path
	frame: int
	true: int
	predicted: Optional[int]
	reason: Optional[str]
	raw: str
	latency: float

	@property
	def correct(self) -> bool:
		return self.predicted is not None and self.predicted == self.true


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Evaluate GPT-4o on Robovie frames.")
	parser.add_argument("--image-root", type=Path, default=Path("images"))
	parser.add_argument(
		"--trimmed-root",
		type=Path,
		default=Path("images_trimmed"),
		help="Directory for cropped images (created on first run).",
	)
	parser.add_argument("--pause", type=float, default=0.0, help="Sleep seconds between requests.")
	parser.add_argument("--output-json", type=Path, default=Path("results.json"))
	parser.add_argument("--frame-plot", type=Path, default=Path("frame_accuracy.png"))
	parser.add_argument(
		"--latency-plot",
		type=Path,
		default=Path("response_times.png"),
		help="Line plot of per-request response times.",
	)
	parser.add_argument("--limit", type=int, default=None, help="Send only the first N images.")
	return parser.parse_args()


def build_client() -> OpenAI:
	load_dotenv()
	api_key = os.getenv("OPENAI_API_KEY")
	if not api_key:
		raise EnvironmentError("OPENAI_API_KEY is missing. Check your .env file.")
	return OpenAI(api_key=api_key)


def list_images(root: Path) -> list[Path]:
	if not root.exists():
		raise FileNotFoundError(f"Image directory not found: {root}")
	paths = sorted(root.glob("*/*.png"))
	if not paths:
		raise FileNotFoundError(f"No PNG images found under {root}")
	return paths


def append_suffix(path: Path, suffix: str) -> Path:
	if not suffix:
		return path
	return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def reset_output_dir(root: Path) -> None:
	if root.exists():
		shutil.rmtree(root)
	root.mkdir(parents=True, exist_ok=True)


def ensure_trimmed_images(source_root: Path, trimmed_root: Path) -> Path:
	"""Ensure cropped images live under ``trimmed_root / "original"``."""
	trimmed_root.mkdir(parents=True, exist_ok=True)
	original_root = trimmed_root / "original"
	if original_root.exists():
		return original_root

	# Migrate legacy layout where sequences sat directly under ``trimmed_root``.
	legacy_children = [
		path
		for path in trimmed_root.iterdir()
		if path.is_dir() and path.name not in SCALE_LABELS
	]
	if legacy_children:
		original_root.mkdir(parents=True, exist_ok=True)
		for legacy_dir in legacy_children:
			target_dir = original_root / legacy_dir.name
			if target_dir.exists():
				continue
			shutil.move(str(legacy_dir), str(target_dir))
		return original_root

	if not source_root.exists():
		raise FileNotFoundError(f"Source image directory not found: {source_root}")
	original_root.mkdir(parents=True, exist_ok=True)
	for sequence_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
		target_dir = original_root / sequence_dir.name
		target_dir.mkdir(parents=True, exist_ok=True)
		for image_path in sorted(sequence_dir.glob("*.png")):
			with Image.open(image_path) as image:
				cropped = image.crop(CROP_BOX)
				cropped.save(target_dir / image_path.name)
	return original_root


def ensure_scaled_images(trimmed_root: Path, scale: float, label: str) -> Path:
	original_root = trimmed_root / "original"
	if not original_root.exists():
		raise FileNotFoundError(f"Original trimmed images not found: {original_root}")
	if math.isclose(scale, 1.0):
		return original_root

	scaled_root = trimmed_root / label
	legacy_root = trimmed_root.parent / f"{trimmed_root.name}_{label}"
	if scaled_root.exists():
		return scaled_root
	if legacy_root.exists():
		shutil.move(str(legacy_root), str(scaled_root))
		return scaled_root

	for sequence_dir in sorted(path for path in original_root.iterdir() if path.is_dir()):
		target_dir = scaled_root / sequence_dir.name
		target_dir.mkdir(parents=True, exist_ok=True)
		for image_path in sorted(sequence_dir.glob("*.png")):
			with Image.open(image_path) as image:
				new_size = (
					max(1, int(round(image.width * scale))),
					max(1, int(round(image.height * scale))),
				)
				resized = image.resize(new_size, Image.LANCZOS)
				target_path = target_dir / image_path.name
				resized.save(target_path)
	return scaled_root


def expected_label(path: Path) -> int:
	if not path.name[0].isdigit():
		raise ValueError(f"Cannot infer label from file name: {path.name}")
	return 1 if int(path.name[0]) % 2 == 0 else 0


def frame_index(path: Path) -> int:
	try:
		return int(path.stem.split("_")[1])
	except (IndexError, ValueError) as error:
		raise ValueError(f"Cannot parse frame index from {path.name}") from error


def to_data_url(path: Path) -> str:
	with path.open("rb") as handle:
		payload = base64.b64encode(handle.read()).decode("ascii")
	return f"data:image/png;base64,{payload}"


def call_gpt(client: OpenAI, image_path: Path) -> str:
	image_data = to_data_url(image_path)
	response = client.responses.create(
		model="gpt-5.1",
		store=False,
		reasoning={"effort": "low"},
		text={"verbosity": "low"},
		input=[
			{
				"role": "user",
				"content": [
					{"type": "input_text", "text": PROMPT_TEXT},
					{"type": "input_image", "image_url": image_data},
				],
			}
		],
		max_output_tokens=200,
	)
	return response.output_text.strip()


def safe_predict(client: OpenAI, image_path: Path, retries: int = 4) -> tuple[str, float]:
	delay = 2.0
	start = time.perf_counter()
	for attempt in range(1, retries + 1):
		try:
			response = call_gpt(client, image_path)
			elapsed = time.perf_counter() - start
			return response, elapsed
		except Exception as error:  # noqa: BLE001
			if attempt == retries:
				raise RuntimeError(f"Failed to call GPT for {image_path}") from error
			print(f"Retry {attempt}/{retries - 1} for {image_path.name}: {error}. Waiting {delay:.1f}s")
			time.sleep(delay)
			delay *= 2


def parse_response(raw: str) -> tuple[Optional[int], Optional[str]]:
	raw = raw.strip()
	start, end = raw.find("{"), raw.rfind("}")
	if start == -1 or end == -1 or start >= end:
		return None, None

	try:
		payload = json.loads(raw[start : end + 1])
	except json.JSONDecodeError:
		return None, None

	label = payload.get("label")
	reason = payload.get("reason")
	if isinstance(label, int) and label in (0, 1):
		return label, reason if isinstance(reason, str) else None
	return None, reason if isinstance(reason, str) else None


def iterate_results(client: OpenAI, paths: list[Path], pause: float) -> list[Result]:
	results: list[Result] = []
	for idx, path in enumerate(paths, start=1):
		truth = expected_label(path)
		frame = frame_index(path)
		raw, latency = safe_predict(client, path)
		pred, reason = parse_response(raw)

		results.append(
			Result(path=path, frame=frame, true=truth, predicted=pred, reason=reason, raw=raw, latency=latency)
		)

		print(
			f"[{idx}] {path} -> truth {truth}, predicted {pred}, correct {pred == truth}, "
			f"latency {latency:.2f}s"
		)
		if pause:
			time.sleep(pause)
	return results


def summarise(
	results: list[Result],
) -> tuple[float, dict[int, float], dict[int, float], dict[int, dict[int, float]]]:
	total = len(results)
	if total == 0:
		return 0.0, {}, {}, {}

	correct_total = sum(r.correct for r in results)

	per_label_totals: Counter[int] = Counter(r.true for r in results)
	per_label_correct: Counter[int] = Counter(r.true for r in results if r.correct)

	per_frame_flags: defaultdict[int, list[bool]] = defaultdict(list)
	per_frame_label_totals: defaultdict[int, Counter[int]] = defaultdict(Counter)
	per_frame_label_correct: defaultdict[int, Counter[int]] = defaultdict(Counter)
	for result in results:
		per_frame_flags[result.frame].append(result.correct)
		per_frame_label_totals[result.frame][result.true] += 1
		if result.correct:
			per_frame_label_correct[result.frame][result.true] += 1

	per_label_accuracy = {
		label: per_label_correct[label] / count if count else 0.0
		for label, count in per_label_totals.items()
	}

	frame_accuracy = {
		frame: sum(flags) / len(flags) if flags else 0.0
		for frame, flags in sorted(per_frame_flags.items())
	}

	frame_label_accuracy: dict[int, dict[int, float]] = {}
	for frame in sorted(per_frame_label_totals.keys()):
		label_totals = per_frame_label_totals[frame]
		label_correct = per_frame_label_correct[frame]
		frame_label_accuracy[frame] = {
			label: (label_correct[label] / total if total else 0.0)
			for label, total in label_totals.items()
		}

	return correct_total / total, per_label_accuracy, frame_accuracy, frame_label_accuracy


def save_results(results: list[Result], path: Path) -> None:
	serialised = [
		{
			"image_path": str(r.path),
			"frame_index": r.frame,
			"true_label": r.true,
			"predicted_label": r.predicted,
			"reason": r.reason,
			"raw_response": r.raw,
			"correct": r.correct,
			"response_time_seconds": r.latency,
		}
		for r in results
	]
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(serialised, ensure_ascii=False, indent=2), encoding="utf-8")


def save_frame_plot(
	frame_accuracy: dict[int, float],
	frame_label_accuracy: dict[int, dict[int, float]],
	path: Path,
) -> None:
	if not frame_accuracy:
		return

	frames = sorted(frame_accuracy.keys())
	values = [frame_accuracy[f] for f in frames]

	plt.figure(figsize=(10, 4))
	plt.plot(frames, values, marker="o", label="Overall")
	label_ids = sorted({label for data in frame_label_accuracy.values() for label in data})
	for label in label_ids:
		label_values = [
			frame_label_accuracy.get(frame, {}).get(label, math.nan)
			for frame in frames
		]
		plt.plot(frames, label_values, marker="o", label=f"Label {label}")
	plt.ylim(0, 1)
	plt.xlabel("Frame index")
	plt.ylabel("Accuracy")
	plt.title("Frame-wise accuracy")
	plt.grid(True, linestyle="--", alpha=0.4)
	plt.xticks(frames)
	plt.legend()
	plt.tight_layout()
	plt.savefig(path, dpi=200)
	plt.close()


def save_label_heatmaps(results: list[Result], output_dir: Path) -> list[Path]:
	output_dir.mkdir(parents=True, exist_ok=True)
	if not results:
		return []

	def trial_id(result: Result) -> str:
		prefix = result.path.stem.split("_")[0]
		participant = result.path.parent.name
		return f"{participant}-{prefix}"

	frames = sorted({result.frame for result in results})
	saved_paths: list[Path] = []
	for label in sorted({result.true for result in results}):
		label_results = [result for result in results if result.true == label]
		if not label_results:
			continue

		trials = sorted({trial_id(result) for result in label_results})
		value_map = {
			(trial_id(result), result.frame): (
				float(result.predicted) if result.predicted is not None else math.nan
			)
			for result in label_results
		}

		matrix: list[list[float]] = []
		for trial in trials:
			row: list[float] = []
			for frame in frames:
				row.append(value_map.get((trial, frame), math.nan))
			matrix.append(row)

		fig, ax = plt.subplots(figsize=(12, max(3, len(trials) * 0.4)))
		im = ax.imshow(matrix, aspect="auto", interpolation="nearest", vmin=0, vmax=1, cmap="viridis")
		ax.set_xlabel("Frame index")
		ax.set_ylabel("Trial")
		ax.set_title(f"Predicted label heatmap (true label {label})")
		ax.set_xticks(range(len(frames)))
		ax.set_xticklabels([f"{frame:02d}" for frame in frames], rotation=45)
		ax.set_yticks(range(len(trials)))
		ax.set_yticklabels(trials)
		fig.colorbar(im, ax=ax, label="Predicted label")
		fig.tight_layout()
		output_path = output_dir / f"label_{label}_heatmap.png"
		fig.savefig(output_path, dpi=200)
		plt.close(fig)
		saved_paths.append(output_path)

	return saved_paths


def save_latency_plot(latencies: list[float], path: Path, title: str = "OpenAI response times") -> None:
	if not latencies:
		return
	path.parent.mkdir(parents=True, exist_ok=True)
	indices = list(range(1, len(latencies) + 1))
	plt.figure(figsize=(10, 4))
	plt.plot(indices, latencies, marker="o")
	plt.xlabel("Request index")
	plt.ylabel("Response time (s)")
	plt.title(title)
	plt.grid(False)
	ax = plt.gca()
	max_latency = max(latencies)
	ax.set_ylim(0, max_latency * 1.1)
	ax.yaxis.set_major_locator(MaxNLocator(nbins=12, steps=[1, 2, 5, 10]))
	for y in ax.get_yticks():
		ax.axhline(y, color="gray", linestyle=":", linewidth=0.5, alpha=0.6)
	plt.tight_layout()
	plt.savefig(path, dpi=200)
	plt.close()


def save_combined_latency_plot(latency_map: dict[str, list[float]], path: Path) -> None:
	valid = {label: latencies for label, latencies in latency_map.items() if latencies}
	if not valid:
		return
	path.parent.mkdir(parents=True, exist_ok=True)
	plt.figure(figsize=(10, 4))
	for label, latencies in sorted(valid.items()):
		indices = list(range(1, len(latencies) + 1))
		plt.plot(indices, latencies, marker="o", label=label)
	plt.xlabel("Request index")
	plt.ylabel("Response time (s)")
	plt.title("Response times (all scales)")
	plt.grid(False)
	ax = plt.gca()
	all_latencies = [value for latencies in valid.values() for value in latencies]
	ax.set_ylim(0, max(all_latencies) * 1.1)
	ax.yaxis.set_major_locator(MaxNLocator(nbins=12, steps=[1, 2, 5, 10]))
	for y in ax.get_yticks():
		ax.axhline(y, color="gray", linestyle=":", linewidth=0.5, alpha=0.6)
	plt.legend()
	plt.tight_layout()
	plt.savefig(path, dpi=200)
	plt.close()


def main() -> None:
	args = parse_args()
	reset_output_dir(OUTPUT_DIR)
	ensure_trimmed_images(args.image_root, args.trimmed_root)
	client = build_client()

	base_results = OUTPUT_DIR / args.output_json.name
	base_frame_plot = OUTPUT_DIR / args.frame_plot.name
	base_latency_plot = OUTPUT_DIR / args.latency_plot.name
	report_path = OUTPUT_DIR / "summary.txt"
	report_lines: list[str] = []
	combined_latencies: dict[str, list[float]] = {}

	for scale, label, suffix in SCALE_CONFIGS:
		scaled_root = ensure_scaled_images(args.trimmed_root, scale, label)
		paths = list_images(scaled_root)
		if args.limit is not None:
			paths = paths[: args.limit]

		print(f"\n=== Evaluation for {label} (scale ×{scale:.3f}) ===")
		report_lines.append(f"=== Evaluation for {label} (scale ×{scale:.3f}) ===")
		results = iterate_results(client, paths, pause=args.pause)
		overall, per_label, frame_accuracy, frame_label_accuracy = summarise(results)

		sum_correct = sum(r.correct for r in results)
		total = len(results)
		print(f"Overall: {overall:.3%} ({sum_correct}/{total})")
		report_lines.append(f"Overall: {overall:.3%} ({sum_correct}/{total})")
		for lbl in sorted(per_label):
			txt = f"Label {lbl}: {per_label[lbl]:.3%}"
			print(txt)
			report_lines.append(txt)

		print("Frame accuracy:")
		report_lines.append("Frame accuracy:")
		for frame in sorted(frame_accuracy):
			accuracy = frame_accuracy[frame]
			frame_line = f"  Frame {frame:02d}: {accuracy:.3%}"
			print(frame_line)
			report_lines.append(frame_line)
			label_breakdown = frame_label_accuracy.get(frame, {})
			for lbl in sorted(label_breakdown):
				lbl_line = f"    Label {lbl}: {label_breakdown[lbl]:.3%}"
				print(lbl_line)
				report_lines.append(lbl_line)

		latencies = [result.latency for result in results]
		if latencies:
			average_latency = statistics.mean(latencies)
			latency_line = f"Average response time: {average_latency:.2f}s over {len(latencies)} requests"
		else:
			latency_line = "Average response time: n/a (no results)"
		print(latency_line)
		report_lines.append(latency_line)
		combined_latencies[label] = latencies

		results_path = append_suffix(base_results, suffix)
		frame_plot_path = append_suffix(base_frame_plot, suffix)
		latency_plot_path = append_suffix(base_latency_plot, suffix)
		heatmap_dir = HEATMAP_DIR / label

		save_results(results, results_path)
		save_frame_plot(frame_accuracy, frame_label_accuracy, frame_plot_path)
		save_latency_plot(latencies, latency_plot_path, title=f"Response times ({label})")
		heatmap_paths = save_label_heatmaps(results, heatmap_dir)

		saved_lines = [
			f"Saved details to {results_path}",
			f"Saved frame plot to {frame_plot_path}",
		]
		for message in saved_lines:
			print(message)
			report_lines.append(message)
		if latencies:
			latency_save = f"Saved latency plot to {latency_plot_path}"
			print(latency_save)
			report_lines.append(latency_save)
		else:
			no_latency = "Latency plot not generated (no results)."
			print(no_latency)
			report_lines.append(no_latency)
		if heatmap_paths:
			for path in heatmap_paths:
				heat_line = f"Saved label heatmap to {path}"
				print(heat_line)
				report_lines.append(heat_line)
		else:
			no_heat = "No heatmaps generated (no results)."
			print(no_heat)
			report_lines.append(no_heat)
		report_lines.append("")

	combined_path = base_latency_plot.with_name(f"{base_latency_plot.stem}_combined{base_latency_plot.suffix}")
	save_combined_latency_plot(combined_latencies, combined_path)
	if any(latencies for latencies in combined_latencies.values()):
		combined_msg = f"Saved combined latency plot to {combined_path}"
		print(f"\n{combined_msg}")
		report_lines.append(combined_msg)
	else:
		no_combined = "Combined latency plot not generated (no results)."
		print(f"\n{no_combined}")
		report_lines.append(no_combined)

	report_lines.append("")
	summary_msg = f"Summary saved to {report_path}"
	print(summary_msg)
	report_lines.append(summary_msg)
	report_path.write_text("\n".join(report_lines).strip() + "\n", encoding="utf-8")


if __name__ == "__main__":
	main()