"""Submit Robovie camera frames to GPT-4o and collect accuracy metrics."""

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
from dotenv import load_dotenv
from openai import OpenAI


PROMPT_TEXT = (
	"The received image is a 360-degree capture from an omnidirectional camera "
	"(RICOH THETA Z1) mounted on top of a humanoid robot called Robovie 2. The "
	"center of the image corresponds to the robot's forward direction; the left "
	"and right edges correspond to the rear; the top and bottom edges correspond "
	"to the upward and downward directions. The yellow objects that appear "
	"symmetrically in the lower left and right are Robovie's eyes.\n\n"
	"Determine whether the human behavior in the footage is an \"intentional "
	"malicious action whose primary goal is to obstruct the robot's path\" or a "
	"\"non-malicious action whose primary goal is not to obstruct the robot's path\". "
	"After making the judgment, briefly output the decision and its rationale, "
	"referring to the following information.\n"
	"- Use label 1 for malicious actions and label 0 for non-malicious actions.\n"
	"- Provide reason as a concise English sentence that explains the rationale.\n\n"
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


def load_positive_labels(root: Path) -> dict[Path, set[str]]:
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


def expected_label(path: Path, positive_map: dict[Path, set[str]]) -> int:
	positives = positive_map.get(path.parent)
	if positives is None:
		raise ValueError(f"No label set loaded for directory: {path.parent}")
	return 1 if path.stem in positives else 0


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


def iterate_results(
	client: OpenAI,
	paths: list[Path],
	pause: float,
	positive_map: dict[Path, set[str]],
) -> list[Result]:
	results: list[Result] = []
	for idx, path in enumerate(paths, start=1):
		truth = expected_label(path, positive_map)
		frame = frame_index(path)
		raw = safe_predict(client, path)
		pred, reason = parse_response(raw)

		results.append(Result(path=path, frame=frame, true=truth, predicted=pred, reason=reason, raw=raw))

		print(f"[{idx}] {path} -> truth {truth}, predicted {pred}, correct {pred == truth}")
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
	plt.xlabel("Frame index (1-20)")
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

	paths = list_images(args.image_root)
	positive_map = load_positive_labels(args.image_root)
	if args.limit is not None:
		paths = paths[: args.limit]

	results = iterate_results(client, paths, pause=args.pause, positive_map=positive_map)

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