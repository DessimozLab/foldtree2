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
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("msa", type=Path, help="raxml-mapped alignment fasta (gaps '-')")
    parser.add_argument("tree", type=Path, help="newick tree with the same tip names")
    parser.add_argument("charmaps", type=Path, help="charmaps pickle from makesubmat.py")
    parser.add_argument("submat", type=Path, help="*_submat.txt (RAxML-NG custom GTR format)")
    parser.add_argument("--arpip", type=Path, default=Path("ARPIP"), help="ARPIP binary")
    parser.add_argument("--lambda", type=float, default=10.0, dest="lambda_", help="indel rate")
    parser.add_argument("--mu", type=float, default=0.01, help="gap extension rate")
    parser.add_argument("--out", type=Path, default=None, help="output directory (default: <msa stem>_arpip)")
    parser.add_argument("--native", action="store_true",
                        help="use the raw encoder charset instead of the raxml-mapped one")
    args = parser.parse_args()

    _, char_set, _, raxml_charset, _ = pickle.load(open(args.charmaps, "rb"))
    charset = char_set if args.native else raxml_charset
    if len(charset) > 64:
        raise SystemExit(f"charset has {len(charset)} characters; raxml-ng/ARPIP support at most 64")

    out = args.out or Path(str(args.msa.with_suffix("")) + "_arpip")
    out.mkdir(parents=True, exist_ok=True)
    charset_file = out / "charset.txt"
    charset_file.write_text("\n".join(charset) + "\n")
    conf = out / "conf.txt"
    conf.write_text(f"""analysis_name=arpip_asr
alphabet=Custom(file={charset_file.resolve()})
opt.seed=1
opt.likelihood=0
opt.pip_param_estimate=0
opt.combine_msa_asr=1
input.sequence.file={args.msa.resolve()}
input.sequence.sites_to_use=all
init.tree=user
input.tree.file={args.tree.resolve()}
model=PIP(model=GTR(file={args.submat.resolve()}),lambda={args.lambda_}, mu={args.mu})
rate_distribution=Constant
output.tree.file={out / 'tree.nwk'}
output.ancestral.file={out / 'anc.fasta'}
output.node_rel.file={out / 'node_rel.txt'}
output.mlindelpoints.file={out / 'mlindelpoints.txt'}
output.ancestral.prob_profile.file={out / 'asr_prob.txt'}
opt.ancestral.prob_profile=normalized
""")
    subprocess.run([str(args.arpip), f"params={conf}"], check=True)
    print(f"Ancestral sequences: {out / 'anc.fasta'}")
    print(f"Posterior profile:   {out / 'asr_prob.txt'}")


if __name__ == "__main__":
    main()
