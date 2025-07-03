import sys
from ft2treebuilder import treebuilder

if __name__ == "__main__":
    # Usage: python scripts/replace_sp_chars.py <input_fasta> <output_fasta>
    if len(sys.argv) < 3:
        print("Usage: python replace_sp_chars.py <input_fasta> <output_fasta>")
        sys.exit(1)
    input_fasta = sys.argv[1]
    output_fasta = sys.argv[2]
    tb = treebuilder(model="models/model")  # Dummy model path, not used for this function
    tb.replace_sp_chars(encoded_fasta=input_fasta, outfile=output_fasta)
