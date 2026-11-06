"""Run ARPIP (bpp-ARPIP) ancestral reconstruction on a foldtree2 alignment.

Reuses what foldtree2 already produces: the charmaps pickle gives the raxml
charset and the *_submat.txt matrix is passed to ARPIP unchanged (it is
already in the RAxML-NG custom GTR format). Requires an ARPIP binary built
from https://github.com/DessimozLab/bpp-ARPIP (branch feature/asr-integration).

Example:
    python scripts/ancestral/arpip_asr.py run/raxml_alignment.fasta run/bestTree.nwk \
        models/production/10char_minimal_decoder/production_10char_best_encoder_pair_counts.pkl \
        models/production/10char_minimal_decoder/production_10char_best_encoder_submat.txt \
        --arpip /path/to/ARPIP --lambda 10 --mu 0.01
"""

import argparse
import pickle
from pathlib import Path

from foldtree2.src.trees.arpip import run_arpip_asr


def load_charmaps(charmaps_pkl: Path):
    """Return (char_set, raxml_charset) from a charmaps pickle."""
    _, char_set, _, raxml_charset, _ = pickle.load(open(charmaps_pkl, "rb"))
    return char_set, raxml_charset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("msa", type=Path, help="raxml-mapped alignment fasta (gaps '-')")
    parser.add_argument("tree", type=Path, help="newick tree with the same tip names")
    parser.add_argument("charmaps", type=Path, help="charmaps pickle from makesubmat.py")
    parser.add_argument("submat", type=Path, help="*_submat.txt (RAxML-NG custom GTR format)")
    parser.add_argument("--arpip", type=Path, default=Path("ARPIP"), help="ARPIP binary")
    parser.add_argument("--lambda", type=float, default=10.0, dest="indel_lambda", help="indel rate")
    parser.add_argument("--mu", type=float, default=0.01, help="gap extension rate")
    parser.add_argument("--out", type=Path, default=None, help="output prefix (default: <msa stem>.arpip)")
    parser.add_argument("--native", action="store_true",
                        help="use the raw encoder charset instead of the raxml-mapped one")
    args = parser.parse_args()

    char_set, raxml_charset = load_charmaps(args.charmaps)
    prefix = args.out or Path(str(args.msa.with_suffix("")) + ".arpip")
    outputs = run_arpip_asr(args.msa, args.tree,
                            char_set if args.native else raxml_charset, args.submat, prefix,
                            arpip=args.arpip, indel_lambda=args.indel_lambda, indel_mu=args.mu)
    print(f"Ancestral sequences: {outputs['ancestral']}")
    print(f"Posterior profile:   {outputs['probabilities']}")


if __name__ == "__main__":
    main()
