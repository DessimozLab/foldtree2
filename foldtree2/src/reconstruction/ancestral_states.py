"""Ancestral ft2 states and their decoded amino acids for an OMA family.

The ft2 alphabet is structural, so RAxML-NG marginal ASR returns a structural
state for every column but no residue identity: the treebuilder decoder maps
states back to amino acids.  ASR is run on the same MAFFT ft2 alignment that
fed the tree, so ancestral rows arrive already in that alignment's column
layout.  The one realignment still needed is restoring gaps - marginal ASR
assigns a state to every column, and a column is a gap for a node only where
none of its descendant leaves carry a residue.

Outputs (written to the family directory):
    ancestral_ft2.fasta   internal node -> ft2 states, gaps, alignment columns
    ancestral_aa.fasta    internal node -> decoded amino acids, same columns
"""
from __future__ import annotations

import argparse
import functools
from pathlib import Path
from typing import Dict, List, Optional

import ete3
import torch

from foldtree2.src.reconstruction.tree_msa_dataset import (
    ANCESTRAL_AA, ANCESTRAL_FT2, family_ft2_alignment, read_fasta, _strip_mafft_sentinel)
from foldtree2.src.trees.ancestral import run_ancestral


def tree_for(alignment: Path) -> Path:
    """The RAxML tree inferred from `alignment` (same stem, no _aln.fasta)."""
    stem = str(alignment).split(".raxml_aln.fasta")[0]
    tree = Path(stem + ".raxml.bestTree")
    if not tree.is_file():
        raise FileNotFoundError(f"No RAxML tree for alignment {alignment}")
    return tree


def fitted_model(alignment: Path) -> str:
    """The substitution model RAxML fitted to `alignment`, as a model string."""
    stem = str(alignment).split(".raxml_aln.fasta")[0]
    path = Path(stem + ".raxml.bestModel")
    if not path.is_file():
        raise FileNotFoundError(f"No fitted RAxML model for alignment {alignment}")
    return path.read_text().split(", noname")[0].strip()


def raxml_symbols(builder) -> Dict[str, str]:
    """ft2 alignment char -> RAxML-safe symbol."""
    return {char: builder.raxmlchars[builder.map[char]] for char in builder.alphabet}


def raxml_to_ft2(builder) -> Dict[str, str]:
    """RAxML-safe symbol -> the ft2 character stored in the alignment."""
    return {sym: char for char, sym in raxml_symbols(builder).items()}


def decoder_ids(builder) -> Dict[str, int]:
    """RAxML-safe symbol -> embedding/decoder index of that ft2 state.

    ``builder.alphabet`` is sorted, so its position is not the state id; the id
    is the ordinal of the character before the RAxML-safe substitution.
    """
    return {sym: ord(builder.rev_replace_dict.get(char, char)) - 1
            for char, sym in raxml_symbols(builder).items()}


def write_raxml_alignment(sequences: Dict[str, str], symbols: Dict[str, str], out: Path) -> Path:
    with open(out, "w", encoding="latin-1") as fh:
        for key, seq in sequences.items():
            fh.write(f">{key}\n{''.join(symbols.get(c, '-') for c in seq)}\n")
    return str(out)


def read_states(path: Path) -> Dict[str, str]:
    """RAxML-NG ancestralStates: one TAB-separated node<TAB>sequence per line."""
    states = {}
    for line in Path(path).read_text(encoding="latin-1").split("\n"):
        if not line.strip():
            continue
        node, _, seq = line.partition("\t")
        states[node.strip()] = seq
    return states


def gap_mask(ancestral: str, leaf_sequences: List[str]) -> List[bool]:
    """Keep a column iff some descendant leaf of the node has a residue there."""
    return [any(seq[c] != "-" for seq in leaf_sequences) for c in range(len(ancestral))]


def gapped(sequence: str, mask: List[bool]) -> str:
    return "".join(c if keep else "-" for c, keep in zip(sequence, mask))


def decode(builder, ids: Dict[str, int], sequence: str) -> str:
    """Decode the contiguous chain of non-gap ft2 states into amino acids."""
    ords = [ids[c] for c in sequence if c != "-"]
    if not ords:
        return ""
    return builder.decoder_reconstruction(torch.tensor(ords, dtype=torch.long))["aastr"]


@functools.lru_cache(maxsize=4)
def _builder(encoder: str, decoder: str):
    """Cache the treebuilder so a batch of families loads the models once."""
    from foldtree2.ft2treebuilder import treebuilder
    return treebuilder(encoder, decoder_model=decoder, verbosity=0)


def reconstruct(family_dir: Path, encoder: Path, decoder: Path, *,
                alignment: Optional[Path] = None, tree: Optional[Path] = None,
                model: Optional[str] = None, raxml_ng: str = "raxml-ng",
                outdir: Optional[Path] = None, threads: Optional[int] = None,
                overwrite: bool = False) -> Dict[str, str]:
    """Run ASR on a family and return node -> gapped ft2 states.

    The decoded amino acids are written alongside in `ANCESTRAL_AA`.  Returns
    the ft2 rows; read them back with `load_ancestral` if only the files are
    wanted.
    """
    family = Path(family_dir)
    outdir = Path(outdir) if outdir else family
    outdir.mkdir(parents=True, exist_ok=True)
    ft2_path = Path(alignment) if alignment else family_ft2_alignment(family)
    tree_path = Path(tree) if tree else tree_for(ft2_path)
    model = model or fitted_model(ft2_path)

    tb = _builder(str(encoder), str(decoder))
    symbols = raxml_symbols(tb)
    ascii_symbols = raxml_to_ft2(tb)
    ids = decoder_ids(tb)

    # Sentinel-stripped so the ancestral rows share the column layout the
    # geometry dataset uses.
    sequences = {t: _strip_mafft_sentinel(s) for t, s in read_fasta(ft2_path).items()}
    rax_aln = outdir / "ancestral_asr.fasta"
    write_raxml_alignment(sequences, symbols, rax_aln)

    outputs = run_ancestral(rax_aln, tree_path, model, str(outdir / "ancestral"),
                           executable=raxml_ng, threads=threads, overwrite=overwrite)
    states = read_states(outputs["states"])
    # ete3 refuses the RAxML-NG ancestral tree when handed a path.
    rooted = ete3.Tree(Path(outputs["tree"]).read_text(), format=1)

    ft2: Dict[str, str] = {}
    aa: Dict[str, str] = {}
    for node in rooted.traverse():
        name = node.name or "root"
        if not node.children:          # ete3 exposes is_leaf as a method here
            continue
        seq = states.get(name)
        # ancestralStates holds only internal nodes; the tips come from the MSA.
        leaves = [sequences[l.name] for l in node.iter_leaves() if l.name in sequences]
        if seq is None or not leaves:
            continue
        mask = gap_mask(seq, leaves)
        ft2[name] = gapped("".join(ascii_symbols.get(c, "-") for c in seq), mask)
        aa[name] = gapped(decode(tb, ids, seq), mask)

    write_fasta(outdir / ANCESTRAL_FT2, ft2)
    write_fasta(outdir / ANCESTRAL_AA, aa)
    return ft2


def write_fasta(path: Path, records: Dict[str, str]) -> Path:
    with open(path, "w", encoding="latin-1") as fh:
        for key, seq in records.items():
            fh.write(f">{key}\n{seq}\n")
    return Path(path)


def load_ancestral(family_dir: Path) -> Dict[str, str]:
    """Return node -> ft2 states for a family, or {} if none was reconstructed."""
    path = Path(family_dir) / ANCESTRAL_FT2
    if not path.is_file():
        return {}
    return read_fasta(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family-dir", required=True)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--decoder", required=True)
    ap.add_argument("--alignment")
    ap.add_argument("--tree")
    ap.add_argument("--model")
    ap.add_argument("--raxml-ng", default="raxml-ng")
    ap.add_argument("--outdir")
    ap.add_argument("--threads", type=int)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    ft2 = reconstruct(args.family_dir, args.encoder, args.decoder,
                      alignment=args.alignment, tree=args.tree, model=args.model,
                      raxml_ng=args.raxml_ng, outdir=args.outdir,
                      threads=args.threads, overwrite=args.overwrite)
    length = len(next(iter(ft2.values())))
    print(f"{len(ft2)} internal nodes, {length} columns -> "
          f"{Path(args.outdir or args.family_dir) / ANCESTRAL_FT2}")


if __name__ == "__main__":
    main()