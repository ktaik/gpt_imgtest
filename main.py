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

import matplotlib.pyplot as plt
from dotenv import load_dotenv
from openai import OpenAI


HEATMAP_DIR = Path("plots")

PROMPT_TEXT = (
	"The received image is a 360-degree photo captured by an omnidirectional camera mounted on top of the humanoid robot Robovie 2. The central half of the width corresponds to the robot's forward direction, each outer quarter on the left and right represents the rear direction, and the upper and lower edges correspond to the upward and downward directions. The yellow objects that appear symmetrically near the lower left and right edges are Robovie's eyes.\n\n"
	"Determine whether the person’s behavior in the image is an \"intentional malicious action whose primary goal is to obstruct the robot’s path\" or a \"non-malicious action whose primary goal is not to obstruct the robot’s path\". When judging, remember that merely standing in front of the robot without showing malicious characteristics should not be considered malicious even if it blocks the path. Likewise, people standing beside or behind the robot—not directly in front—should be treated as non-malicious. After making your judgment, use the following guidance to briefly output the decision and its rationale.\n"
	"- Assign label 1 to malicious actions and label 0 to non-malicious actions.\n"
	"- Provide the reason as a concise English sentence that explains the key evidence.\n\n"
	"Examples characteristic of malicious actions include:\n"
	"- Spreading both arms in front of the robot\n"
	"- Spreading feet in front of the robot\n"
	"- Extending a foot in front of the robot\n"
	"- Stopping in front of the robot without facing it\n"
	"- Moving sideways in front of the robot like a crab walk\n"
	"- Hitting, punching, or pushing the robot\n"
	"- Continuously blindfolding the robot\n\n"
	"Selected examples characteristic of non-malicious actions include:\n"
	"- Peering at the robot’s face or body from the side or diagonally in front\n"
	"- Walking alongside the robot from behind, beside, or diagonally in front at a matching speed\n"
	"- Calmly remaining in front of the robot without malicious indicators, even if the path is briefly blocked\n"
	"- Gently touching or stroking the robot\n\n"
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

	@property
	def correct(self) -> bool:
		return self.predicted is not None and self.predicted == self.true


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Evaluate GPT-4o on Robovie frames.")
	parser.add_argument("--image-root", type=Path, default=Path("images"))
	parser.add_argument("--pause", type=float, default=0.0, help="Sleep seconds between requests.")
	parser.add_argument("--output-json", type=Path, default=Path("results.json"))
	parser.add_argument("--frame-plot", type=Path, default=Path("frame_accuracy.png"))
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
		model="gpt-4o",
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


def safe_predict(client: OpenAI, image_path: Path, retries: int = 4) -> str:
	delay = 2.0
	for attempt in range(1, retries + 1):
		try:
			return call_gpt(client, image_path)
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
		raw = safe_predict(client, path)
		pred, reason = parse_response(raw)

		results.append(Result(path=path, frame=frame, true=truth, predicted=pred, reason=reason, raw=raw))

		print(f"[{idx}] {path} -> truth {truth}, predicted {pred}, correct {pred == truth}")
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
		}
		for r in results
	]
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


def main() -> None:
	args = parse_args()
	client = build_client()

	paths = list_images(args.image_root)
	if args.limit is not None:
		paths = paths[: args.limit]

	results = iterate_results(client, paths, pause=args.pause)

	overall, per_label, frame_accuracy, frame_label_accuracy = summarise(results)

	print("\n=== Accuracy summary ===")
	print(f"Overall: {overall:.3%} ({sum(r.correct for r in results)}/{len(results)})")
	for label in sorted(per_label):
		print(f"Label {label}: {per_label[label]:.3%}")

	print("\nFrame accuracy:")
	for frame in sorted(frame_accuracy):
		accuracy = frame_accuracy[frame]
		print(f"Frame {frame:02d}: {accuracy:.3%}")
		label_breakdown = frame_label_accuracy.get(frame, {})
		for label in sorted(label_breakdown):
			print(f"  Label {label}: {label_breakdown[label]:.3%}")

	save_results(results, args.output_json)
	save_frame_plot(frame_accuracy, frame_label_accuracy, args.frame_plot)
	heatmap_paths = save_label_heatmaps(results, HEATMAP_DIR)

	print(f"\nSaved details to {args.output_json}")
	print(f"Saved frame plot to {args.frame_plot}")
	if heatmap_paths:
		for path in heatmap_paths:
			print(f"Saved label heatmap to {path}")
	else:
		print("No heatmaps generated (no results).")


if __name__ == "__main__":
	main()