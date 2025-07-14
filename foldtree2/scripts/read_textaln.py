import sys
from ft2treebuilder import treebuilder

if __name__ == "__main__":
    # Usage: python scripts/read_textaln.py <input_hex> <output_fasta>
    if len(sys.argv) < 3:
        print("Usage: python read_textaln.py <input_hex> <output_fasta>")
        sys.exit(1)
    input_hex = sys.argv[1]
    output_fasta = sys.argv[2]
    tb = treebuilder(model="models/model")  # Dummy model path, not used for this function
    tb.read_textaln(aln_hexfile=input_hex, outfile=output_fasta)
