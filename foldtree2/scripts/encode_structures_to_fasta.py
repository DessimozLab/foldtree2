import argparse
import os
from ft2treebuilder import treebuilder


def resolve_encoder_path(model_arg):
    """Resolve a user-provided model argument to an encoder checkpoint path.

    Accepts either:
    - an explicit encoder file path, or
    - a model prefix (e.g. models/my_model), which maps to modern filenames.
    """
    model_arg = os.path.expanduser(model_arg)

    # Explicit existing file path wins immediately.
    if os.path.isfile(model_arg):
        return model_arg

    # Normalize common suffixes so users can pass either full names or prefixes.
    base = model_arg
    for suffix in ("_best_encoder.pt", "_encoder.pt", ".pt", ".pth"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break

    candidates = [
        f"{base}_best_encoder.pt",
        f"{base}_encoder.pt",
        f"{base}.pt",
        f"{base}.pth",
    ]

    # If only a model name is provided, also try the default models/ directory.
    if os.path.basename(base) == base:
        candidates.extend(
            [
                os.path.join("models", f"{base}_best_encoder.pt"),
                os.path.join("models", f"{base}_encoder.pt"),
                os.path.join("models", f"{base}.pt"),
                os.path.join("models", f"{base}.pth"),
            ]
        )

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(
        "Could not resolve encoder model path from '{arg}'. Tried: {cands}".format(
            arg=model_arg,
            cands=", ".join(candidates),
        )
    )

def main():
    parser = argparse.ArgumentParser(
        description="Encode structures to FASTA using a FoldTree2 encoder model."
    )
    parser.add_argument(
        "model",
        help=(
            "Encoder checkpoint path or model prefix. "
            "Examples: models/my_model_best_encoder.pt or models/my_model"
        ),
    )
    parser.add_argument("structures", help="Glob pattern or directory for input structure files (e.g. '/path/to/structures/*.pdb')")
    parser.add_argument("--outfile", default=None, help="Output FASTA filename (default: encoded.fasta in input directory)")
    parser.add_argument("--n_state", type=int, default=20, help="Number of encoded states/alphabet size")
    args = parser.parse_args()

    model_path = resolve_encoder_path(args.model)
    tb = treebuilder(model=model_path, n_state=args.n_state)
    fasta = tb.encode_structblob(blob=args.structures, outfile=args.outfile)
    print(f"Encoded FASTA written to: {fasta}")

if __name__ == "__main__":
    main()
