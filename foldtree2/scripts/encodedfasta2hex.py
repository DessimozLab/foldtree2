import sys
from ft2treebuilder import treebuilder

if __name__ == "__main__":
    # Usage: python scripts/encodedfasta2hex.py <input_fasta> <output_hex>
    if len(sys.argv) < 3:
        print("Usage: python encodedfasta2hex.py <input_fasta> <output_hex>")
        sys.exit(1)
    input_fasta = sys.argv[1]
    output_hex = sys.argv[2]
    tb = treebuilder(model="models/model")  # Dummy model path, not used for this function
    tb.encodedfasta2hex(encoded_fasta=input_fasta, outfile=output_hex)
