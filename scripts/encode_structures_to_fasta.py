import argparse
import os
from ft2treebuilder import treebuilder

def main():
    parser = argparse.ArgumentParser(description="Encode structures to FASTA using a pickled model.")
    parser.add_argument("model", help="Path to the model (without .pkl extension)")
    parser.add_argument("structures", help="Glob pattern or directory for input structure files (e.g. '/path/to/structures/*.pdb')")
    parser.add_argument("--outfile", default=None, help="Output FASTA filename (default: encoded.fasta in input directory)")
    parser.add_argument("--n_state", type=int, default=20, help="Number of encoded states/alphabet size")
    args = parser.parse_args()

    tb = treebuilder(model=args.model, n_state=args.n_state)
    fasta = tb.encode_structblob(blob=args.structures, outfile=args.outfile)
    print(f"Encoded FASTA written to: {fasta}")

if __name__ == "__main__":
    main()
