"""Export a foldtree2 charset for ARPIP (bpp-ARPIP) ancestral reconstruction.

ARPIP reads custom alphabets with alphabet=Custom(file=<charset>) (one
character per line) and GTR matrices with model=GTR(file=<submat>), using the
same RAxML-NG custom matrix format that makesubmat.py already writes, so the
submat file is passed to ARPIP unchanged.
"""

import argparse
import pickle
from pathlib import Path


def load_charmaps(charmaps_pkl: Path):
    """Return (char_set, raxml_charset) from a charmaps pickle."""
    _, char_set, _, raxml_charset, _ = pickle.load(open(charmaps_pkl, "rb"))
    return char_set, raxml_charset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("charmaps", type=Path, help="charmaps pickle from makesubmat.py")
    parser.add_argument("charset", type=Path, help="output charset file (one character per line)")
    parser.add_argument("--native", action="store_true",
                        help="export the raw encoder charset instead of the raxml-mapped one")
    args = parser.parse_args()

    char_set, raxml_charset = load_charmaps(args.charmaps)
    charset = char_set if args.native else raxml_charset
    args.charset.write_text("\n".join(charset) + "\n")
    print(f"Wrote {len(charset)} characters to {args.charset}")
    print("Run ARPIP with: alphabet=Custom(file=%s) and "
          "model=PIP(model=GTR(file=<submat.txt>),lambda=...,mu=...)" % args.charset)


if __name__ == "__main__":
    main()
