"""Ancestral state reconstruction with the Poisson Indel Process (ARPIP).

ARPIP (https://github.com/DessimozLab/bpp-ARPIP) takes a custom alphabet
(`alphabet=Custom(file=...)`, one character per line) and a GTR matrix in the
RAxML-NG custom format that makesubmat.py already writes, so the foldtree2
`*_submat.txt` is passed unchanged. The alignment and tree are the ones the
treebuilder already produced for RAxML-NG.
"""
from pathlib import Path
import subprocess

# raxml-ng MULTI models accept at most 64 states; ARPIP inherits that limit.
MAX_STATES = 64


def arpip_conf(alignment, tree, charset_file, submat, prefix, *,
               indel_lambda=10.0, indel_mu=0.01):
    """Build the ARPIP parameter file for a foldtree2 alignment and tree."""
    paths = {
        "alphabet": Path(charset_file).resolve(),
        "input.sequence.file": Path(alignment).resolve(),
        "input.tree.file": Path(tree).resolve(),
        "model_submat": Path(submat).resolve(),
        "output.tree.file": str(prefix) + ".arpip.tree",
        "output.ancestral.file": str(prefix) + ".arpip.ancestral.fasta",
        "output.node_rel.file": str(prefix) + ".arpip.noderel",
        "output.mlindelpoints.file": str(prefix) + ".arpip.indelpoints",
        "output.ancestral.prob_profile.file": str(prefix) + ".arpip.probs",
    }
    return "\n".join([
        "analysis_name=arpip_asr",
        f"alphabet=Custom(file={paths['alphabet']})",
        "opt.seed=1",
        "opt.likelihood=0",
        "opt.combine_msa_asr=1",
        f"input.sequence.file={paths['input.sequence.file']}",
        "input.sequence.sites_to_use=all",
        "init.tree=user",
        f"input.tree.file={paths['input.tree.file']}",
        f"model=PIP(model=GTR(file={paths['model_submat']}),"
        f"lambda={indel_lambda}, mu={indel_mu})",
        "rate_distribution=Constant",
        f"output.tree.file={paths['output.tree.file']}",
        f"output.ancestral.file={paths['output.ancestral.file']}",
        f"output.node_rel.file={paths['output.node_rel.file']}",
        f"output.mlindelpoints.file={paths['output.mlindelpoints.file']}",
        f"output.ancestral.prob_profile.file={paths['output.ancestral.prob_profile.file']}",
        "opt.ancestral.prob_profile=normalized",
    ]) + "\n"


def run_arpip_asr(alignment, tree, charset, submat, prefix, *,
                  arpip="ARPIP", indel_lambda=10.0, indel_mu=0.01, runner=None):
    """Write the charset/conf, run ARPIP, and return its output paths.

    `charset` is the alphabet used by `alignment` (the raxml-mapped one for the
    `.raxml_aln.fasta` file). Outputs are named `<prefix>.arpip.*`. ARPIP roots
    an unrooted input tree internally; the input tree is left untouched.
    """
    charset = list(charset)
    if len(set(charset)) != len(charset):
        raise ValueError(f"charset has duplicate characters: {charset}")
    if len(charset) > MAX_STATES:
        raise ValueError(f"charset has {len(charset)} characters; "
                         f"raxml-ng/ARPIP support at most {MAX_STATES}")
    if "-" in charset:
        raise ValueError("'-' is reserved by ARPIP for gaps and cannot be a state")

    prefix = str(prefix)
    Path(prefix).parent.mkdir(parents=True, exist_ok=True)
    charset_file = Path(prefix + ".arpip.charset.txt")
    charset_file.write_text("\n".join(charset) + "\n")
    conf = Path(prefix + ".arpip.conf")
    conf.write_text(arpip_conf(alignment, tree, charset_file, submat, prefix,
                               indel_lambda=indel_lambda, indel_mu=indel_mu))

    command = [str(arpip), f"params={conf}"]
    if runner is None:
        subprocess.run(command, check=True)
    else:
        runner(command)

    return {name: Path(str(prefix) + ".arpip" + suffix)
            for name, suffix in (("ancestral", ".ancestral.fasta"),
                                 ("probabilities", ".probs"),
                                 ("indelpoints", ".indelpoints"),
                                 ("tree", ".tree"),
                                 ("conf", ".conf"),
                                 ("charset", ".charset.txt"))}
