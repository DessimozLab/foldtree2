"""Shared treebuilder-style marginal ancestral reconstruction and FASTA export."""
from pathlib import Path
import subprocess


def ancestral_command(alignment, tree, model, prefix, *, executable="raxml-ng",
                      threads=None, overwrite=False, freeze_fitted=False):
    """Build an ASR command; optionally freeze a strategy's fitted parameters."""
    command = [str(executable), "--ancestral", "--msa", str(alignment),
               "--tree", str(tree), "--model", str(model), "--prefix", str(prefix),
               "--force", "perf_threads"]
    if threads is not None:
        if int(threads) < 1:
            raise ValueError("threads must be positive")
        command.extend(["--threads", str(threads)])
    if freeze_fitted:
        command.extend(["--opt-model", "off", "--opt-branches", "off"])
    if overwrite:
        command.append("--redo")
    return command


def run_ancestral(alignment, tree, model, prefix, *, runner=None, **options):
    """Execute ASR, propagating errors and checking prefix-derived outputs."""
    command = ancestral_command(alignment, tree, model, prefix, **options)
    if runner is None:
        subprocess.run(command, check=True)
    else:
        runner(command)
    outputs = {name: Path(str(prefix) + ".raxml.ancestral" + suffix)
               for name, suffix in (("states", "States"), ("probabilities", "Probs"),
                                    ("tree", "Tree"))}
    for path in outputs.values():
        if not path.is_file():
            raise FileNotFoundError(f"Missing RAxML ancestral output: {path}")
    return outputs


def ancestral_states_to_fasta(source, destination=None):
    """Export native node/tab/sequence records without changing their symbols.

    A 3Di sequence remains 3Di even though its symbols use amino acid letters.
    FT2's optional decoder reconstruction is a separate, FT2-only step.
    """
    source = Path(source)
    destination = Path(destination) if destination is not None else Path(str(source) + ".fasta")
    records, seen = [], set()
    with source.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 2:
                raise ValueError("Malformed ancestral states table")
            identifier, sequence = fields
            if not identifier or any(c.isspace() for c in identifier) or not sequence or any(c.isspace() for c in sequence):
                raise ValueError("Invalid ancestral identifier or sequence")
            if identifier in seen:
                raise ValueError(f"Duplicate ancestral node: {identifier}")
            seen.add(identifier)
            records.append((identifier, sequence))
    if not records or len({len(sequence) for _, sequence in records}) != 1:
        raise ValueError("Empty or unequal-length ancestral sequences")
    if destination.resolve() == source.resolve():
        raise ValueError("FASTA destination must not overwrite the native table")
    with destination.open("w") as stream:
        for identifier, sequence in records:
            stream.write(f">{identifier}\n{sequence}\n")
    return str(destination)
