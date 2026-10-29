# Native tools and token converters

[CLI index](README.md) | [Installation checks](../installation.md#installation-checks) |
[Wrappers](../../foldtree2/mafft_wrappers.py) |
[Source modules](../../foldtree2/src/README.md)

| Command | Role | Distribution |
| --- | --- | --- |
| `mafft` | Align structural tokens using the selected matrix. | Separate dependency; conda recipe installs it. |
| `hex2maffttext`, `maffttext2hex` | Convert token encodings for MAFFT's text mode. | Bundled Linux x86_64 helpers. |
| `raxml-ng` | Multistate maximum-likelihood tree inference. | Bundled custom build; an arbitrary upstream build may lack required support. |
| `mad` | Minimum ancestor deviation rooting. | Bundled Linux x86_64 executable. |
| `foldseek` | Prepare structural reference alignments. | Optional external tool, not needed for basic inference. |

```bash
raxml-ng --version
mafft --version
```

The conversion helpers are stream-oriented native programs, not Python argparse
CLIs. Do not assume `--help` is accepted or modify token FASTA using generic
whitespace stripping: valid alphabets may include control characters.

Native executables require Linux x86_64 with glibc >=2.34 and compatible runtime libraries.
Python installation success does not prove executable compatibility. Pip and
conda use the same package-local resources/wrappers, without a second copy under
the conda prefix's share directory. Existing executable files are not chmod'ed,
so shared/read-only installations do not require ownership merely to execute.

Third-party redistribution/license checks are recorded in the
[release gates](../releasing.md#licensing).
