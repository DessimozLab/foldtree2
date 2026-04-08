import argparse
import os

import torch


def _read_ids(ids_file):
    ids = []
    with open(ids_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            ids.append(line.split()[0])
    return ids


def main():
    parser = argparse.ArgumentParser(
        description='Encode a Foldcomp DB directly to FoldTree2 token FASTA.'
    )
    parser.add_argument('model', type=str, help='Path to trained encoder .pt file')
    parser.add_argument('foldcomp_db', type=str, help='Path to Foldcomp DB basename (without .lookup)')
    parser.add_argument('output_fasta', type=str, help='Output encoded FASTA path')

    parser.add_argument('--device', type=str, default=None, help='Device (e.g., cuda, cuda:0, cpu)')
    parser.add_argument('--ids-file', type=str, default=None, help='Optional text file with Foldcomp IDs (one per line)')
    parser.add_argument('--max-structures', type=int, default=None, help='Optional max number of structures to encode')
    parser.add_argument('--chunk-size', type=int, default=1024, help='Foldcomp prefetch chunk size (default: 1024)')
    parser.add_argument('--queue-size', type=int, default=4, help='Producer/consumer queue size (default: 4)')
    parser.add_argument('--batch-size', type=int, default=16, help='Encoder batch size per chunk (default: 16)')
    parser.add_argument('--cache-size', type=int, default=0, help='Graph cache size in Foldcomp dataset (default: 0)')
    parser.add_argument('--no-replace', action='store_true', help='Disable FASTA special-character replacement')
    parser.add_argument('--quiet', action='store_true', help='Disable progress bar')

    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f'Model not found: {args.model}')

    ids = None
    if args.ids_file is not None:
        ids = _read_ids(args.ids_file)

    device = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = torch.load(args.model, map_location=device, weights_only=False)
    encoder = encoder.to(device)
    encoder.device = device
    encoder.eval()

    output = encoder.encode_foldcomp_fasta(
        foldcomp_db=args.foldcomp_db,
        filename=args.output_fasta,
        ids=ids,
        max_structures=args.max_structures,
        chunk_size=args.chunk_size,
        queue_size=args.queue_size,
        batch_size=args.batch_size,
        cache_size=args.cache_size,
        replace=not args.no_replace,
        verbose=not args.quiet,
    )

    print(f'Encoded FASTA written to: {output}')


if __name__ == '__main__':
    main()
