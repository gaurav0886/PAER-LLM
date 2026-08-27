"""
main.py

Command-line entry point for PAER-LLM.

The original ``main.py`` printed pitch, loudness and spectral features for
``audio_files[0]`` and nothing else - useful as a first smoke test, not as a
project entry point. It also called ``get_audio_files()`` on a dataset
directory containing two copies of the corpus, so ``len(audio_files)`` reported
2880 instead of 1440.

Usage
-----
    python main.py check                       # dataset sanity check
    python main.py features <path/to.wav>      # show extracted features
    python main.py predict <path/to.wav>       # emotion + confidence
    python main.py respond <path/to.wav>       # emotion + LLM reply
"""

from __future__ import annotations

import argparse
import sys


def cmd_check(_args) -> int:
    """Verify the dataset is intact and free of duplicates."""
    from Utils.dataset_loader import get_audio_files, load_dataset_index

    raw = get_audio_files(deduplicate=False)
    index = load_dataset_index()

    print(f"wav files on disk       : {len(raw)}")
    print(f"unique utterances       : {len(index)}")
    print(f"actors                  : {index['actor'].nunique()}")
    print(f"expected (RAVDESS speech): 1440 files, 24 actors")

    if len(raw) != len(index):
        print(
            f"\nWARNING: {len(raw) - len(index)} duplicate files detected.\n"
            "Your Dataset/ folder very likely contains the corpus twice "
            "(e.g. Dataset/Actor_01..24 and Dataset/audio_speech_actors_01-24/"
            "Actor_01..24). Remove the nested copy - duplicates that straddle "
            "a train/test split inflate every metric in the project."
        )

    print("\nClass distribution:")
    print(index["emotion"].value_counts().sort_index().to_string())
    print("\nUtterances per actor:")
    print(index["actor"].value_counts().sort_index().to_string())
    return 0


def cmd_features(args) -> int:
    from Utils.inference import extract_psycho_features
    from Utils.constants import PSYCHO_COLUMNS

    features = extract_psycho_features(args.audio)
    width = max(len(c) for c in PSYCHO_COLUMNS)
    for name, value in zip(PSYCHO_COLUMNS, features):
        print(f"{name:<{width}} : {value: .4f}")
    return 0


def cmd_predict(args) -> int:
    from Utils.inference import predict

    emotion, confidence, embedding, probabilities = predict(args.audio)
    print(f"emotion    : {emotion}")
    print(f"confidence : {confidence:.2f}%")
    print(f"embedding  : {embedding.shape[0]} dims")
    print("\ndistribution:")
    for label, p in sorted(probabilities.items(), key=lambda kv: -kv[1]):
        print(f"  {label:<10} {p:6.2f}%")
    return 0


def cmd_respond(args) -> int:
    from Utils.inference import generate_response

    result = generate_response(args.audio)
    print(f"emotion    : {result['emotion']} ({result['confidence']:.2f}%)")
    print(f"\nresponse:\n{result['response']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="paer-llm", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="dataset sanity check").set_defaults(func=cmd_check)

    for name, func, helptext in (
        ("features", cmd_features, "print psychoacoustic features"),
        ("predict", cmd_predict, "predict emotion"),
        ("respond", cmd_respond, "predict emotion and generate a reply"),
    ):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("audio", help="path to a .wav file")
        p.set_defaults(func=func)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
