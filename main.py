"""Submit stacked Robovie frame triplets to GPT-4o and collect accuracy metrics."""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI


PROMPT_TEXT = (
	"You are given a single composite image made by vertically stacking three consecutive frames captured "
	"at one-second intervals by the omnidirectional RICOH THETA Z1 camera mounted on top of the humanoid "
	"robot Robovie 2. The top segment corresponds to the earliest frame, the middle segment is one second later, "
	"and the bottom segment is two seconds later. In each frame the center faces the robot's forward direction, the "
	"left and right edges correspond to the rear, and the top and bottom edges correspond to the upward and "
	"downward directions. The yellow objects that appear symmetrically near the lower left and right edges are "
	"Robovie's eyes.\n\n"
	"Determine whether any portion of these three frames shows an \"intentional malicious action whose primary goal "
	"is to obstruct the robot's path\" or whether all frames instead show \"non-malicious actions whose primary goal "
	"is not to obstruct the robot's path\". If even a single frame depicts a malicious action, classify the entire "
	"composite as malicious (label 1). Only if all three frames are non-malicious should you return label 0. Provide "
	"a concise English sentence explaining the key evidence and note which part of the stacked image influenced your "
	"decision.\n\n"
	"Examples characteristic of malicious actions include:\n"
	"- Spreading both arms in front of the robot\n"
	"- Spreading feet in front of the robot\n"
	"- Extending a foot in front of the robot\n"
	"- Stopping in front of the robot without facing it\n"
	"- Moving sideways like a crab walk in front of the robot\n"
	"- Hitting, punching, or pushing the robot\n"
	"- Continuously blindfolding the robot\n\n"
	"Examples characteristic of non-malicious actions include:\n"
	"- Peering at the robot's face or body from the side or diagonally in front\n"
	"- Walking alongside the robot from behind, beside, or diagonally in front at a matching speed\n"
	"- Gently touching or stroking the robot\n\n"
	"Respond in JSON format:\n"
	'{"label": <0 or 1>, "reason": "..."}'
)


@dataclass
class Result:
	path: Path
	components: tuple[str, str, str]
	frame: int
	true: int
	predicted: Optional[int]
	reason: Optional[str]
	raw: str

	@property
	def correct(self) -> bool:
		return self.predicted is not None and self.predicted == self.true


@dataclass
class Composite:
	path: Path
	components: tuple[Path, Path, Path]
	frame_start: int
	true_label: int


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Evaluate GPT-4o on Robovie frames.")
	parser.add_argument("--image-root", type=Path, default=Path("images"))
	parser.add_argument("--pause", type=float, default=0.0, help="Sleep seconds between requests.")
	parser.add_argument("--output-json", type=Path, default=Path("results.json"))
	parser.add_argument("--frame-plot", type=Path, default=Path("frame_accuracy.png"))
	parser.add_argument("--limit", type=int, default=None, help="Send only the first N images.")
	parser.add_argument(
		"--sequential-root",
		type=Path,
		default=Path("sequential_images"),
		help="Directory where stacked composites will be stored.",
	)
	return parser.parse_args()


def build_client() -> OpenAI:
	load_dotenv()
	api_key = os.getenv("OPENAI_API_KEY")
	if not api_key:
		raise EnvironmentError("OPENAI_API_KEY is missing. Check your .env file.")
	return OpenAI(api_key=api_key)


def load_positive_labels(root: Path) -> dict[Path, set[str]]:
	if not root.exists():
		raise FileNotFoundError(f"Image directory not found: {root}")
	label_sets: dict[Path, set[str]] = {}
	for subdir in sorted(p for p in root.iterdir() if p.is_dir()):
		label_file = subdir / "label.txt"
		if label_file.exists():
			entries = {
				line.strip()
				for line in label_file.read_text(encoding="utf-8").splitlines()
				if line.strip()
			}
		else:
			entries = set()
		label_sets[subdir] = entries
	return label_sets


def frame_index(path: Path) -> int:
	try:
		return int(path.stem.split("_")[1])
	except (IndexError, ValueError) as error:
		raise ValueError(f"Cannot parse frame index from {path.name}") from error


def stack_images_vertically(paths: tuple[Path, Path, Path]) -> Image.Image:
	images = []
	for image_path in paths:
		with Image.open(image_path) as img:
			images.append(img.convert("RGB"))

	width = max(img.width for img in images)
	total_height = sum(img.height for img in images)
	canvas = Image.new("RGB", (width, total_height))

	current_y = 0
	for img in images:
		if img.width != width:
			img = img.resize((width, img.height))
		canvas.paste(img, (0, current_y))
		current_y += img.height

	return canvas


def generate_composites(
	image_root: Path,
	sequential_root: Path,
	positive_map: dict[Path, set[str]],
) -> list[Composite]:
	sequential_root.mkdir(parents=True, exist_ok=True)
	composites: list[Composite] = []

	for subdir in sorted(p for p in image_root.iterdir() if p.is_dir()):
		if subdir not in positive_map:
			raise ValueError(f"No labels loaded for {subdir}")

		output_dir = sequential_root / subdir.name
		output_dir.mkdir(parents=True, exist_ok=True)
		for leftover in output_dir.glob("*.png"):
			leftover.unlink()

		sequences: defaultdict[str, list[Path]] = defaultdict(list)
		for image_path in sorted(subdir.glob("*.png")):
			prefix = image_path.stem.split("_")[0]
			sequences[prefix].append(image_path)

		for prefix, candidates in sorted(sequences.items()):
			sorted_paths = sorted(candidates, key=frame_index)
			for index in range(len(sorted_paths) - 2):
				window = sorted_paths[index : index + 3]
				indices = [frame_index(path) for path in window]
				if indices[1] != indices[0] + 1 or indices[2] != indices[1] + 1:
					continue

				filename = f"{prefix}_{indices[0]:02d}-{indices[2]:02d}.png"
				output_path = output_dir / filename
				stack_images_vertically(tuple(window)).save(output_path)

				label = 1 if any(path.stem in positive_map[subdir] for path in window) else 0
				composites.append(
					Composite(
						path=output_path,
						components=(window[0], window[1], window[2]),
						frame_start=indices[0],
						true_label=label,
					)
				)

	return sorted(composites, key=lambda item: (item.path.parent, item.path.name))


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


def iterate_results(client: OpenAI, composites: list[Composite], pause: float) -> list[Result]:
	results: list[Result] = []
	for idx, composite in enumerate(composites, start=1):
		raw = safe_predict(client, composite.path)
		pred, reason = parse_response(raw)

		results.append(
			Result(
				path=composite.path,
				components=tuple(path.stem for path in composite.components),
				frame=composite.frame_start,
				true=composite.true_label,
				predicted=pred,
				reason=reason,
				raw=raw,
			)
		)

		print(
			f"[{idx}] {composite.path} -> truth {composite.true_label}, predicted {pred}, "
			f"correct {pred == composite.true_label}"
		)
		if pause:
			time.sleep(pause)
	return results


def summarise(results: list[Result]) -> tuple[float, dict[int, float], dict[int, float]]:
	total = len(results)
	if total == 0:
		return 0.0, {}, {}

	correct_total = sum(r.correct for r in results)

	per_label_totals: Counter[int] = Counter(r.true for r in results)
	per_label_correct: Counter[int] = Counter(r.true for r in results if r.correct)

	per_frame_flags: defaultdict[int, list[bool]] = defaultdict(list)
	for result in results:
		per_frame_flags[result.frame].append(result.correct)

	per_label_accuracy = {
		label: per_label_correct[label] / count if count else 0.0
		for label, count in per_label_totals.items()
	}

	frame_accuracy = {
		frame: sum(flags) / len(flags) if flags else 0.0
		for frame, flags in sorted(per_frame_flags.items())
	}

	return correct_total / total, per_label_accuracy, frame_accuracy


def save_results(results: list[Result], path: Path) -> None:
	serialised = [
		{
			"image_path": str(r.path),
			"components": list(r.components),
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


def save_frame_plot(frame_accuracy: dict[int, float], path: Path) -> None:
	if not frame_accuracy:
		return

	frames = list(frame_accuracy.keys())
	values = [frame_accuracy[f] for f in frames]

	plt.figure(figsize=(10, 4))
	plt.plot(frames, values, marker="o")
	plt.ylim(0, 1)
	plt.xlabel("Starting frame index")
	plt.ylabel("Accuracy")
	plt.title("Frame-wise accuracy")
	plt.grid(True, linestyle="--", alpha=0.4)
	plt.xticks(frames)
	plt.tight_layout()
	plt.savefig(path, dpi=200)
	plt.close()


def main() -> None:
	args = parse_args()
	client = build_client()

	positive_map = load_positive_labels(args.image_root)
	composites = generate_composites(args.image_root, args.sequential_root, positive_map)
	if args.limit is not None:
		composites = composites[: args.limit]

	results = iterate_results(client, composites, pause=args.pause)

	overall, per_label, frame_accuracy = summarise(results)

	print("\n=== Accuracy summary ===")
	print(f"Overall: {overall:.3%} ({sum(r.correct for r in results)}/{len(results)})")
	for label in sorted(per_label):
		print(f"Label {label}: {per_label[label]:.3%}")

	print("\nFrame accuracy:")
	for frame, accuracy in frame_accuracy.items():
		print(f"Frame {frame:02d}: {accuracy:.3%}")

	save_results(results, args.output_json)
	save_frame_plot(frame_accuracy, args.frame_plot)

	print(f"\nSaved details to {args.output_json}")
	print(f"Saved frame plot to {args.frame_plot}")


if __name__ == "__main__":
	main()