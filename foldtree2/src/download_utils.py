"""
Utilities for downloading and managing protein structure datasets.

Handles downloading PDB files from AlphaFold DB and related data sources
with proper multiprocessing support and error handling.
"""

import os
import subprocess
import multiprocessing as mp
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import tqdm


MODEL_VERSION_CANDIDATES = ("v6", "v5", "v4", "v3", "v2")


def _candidate_urls(rep: str) -> List[str]:
    """Return candidate AFDB URLs for a protein ID across known model versions."""
    rep_norm = str(rep).strip().upper()
    return [
        f"https://alphafold.ebi.ac.uk/files/AF-{rep_norm}-F1-model_{version}.pdb"
        for version in MODEL_VERSION_CANDIDATES
    ]


def download_pdb_subprocess(
    rep: str,
    structdir: str,
    timeout: int = 30,
    verbose: bool = False
) -> Optional[str]:
    """
    Download a single PDB file using subprocess to avoid pickling issues.
    
    Args:
        rep: Protein identifier (e.g., 'A0A0C2GTF1')
        structdir: Directory to save the PDB file
        timeout: Timeout for download in seconds
        verbose: Print debug messages
        
    Returns:
        Path to downloaded file if successful, None otherwise
    """
    rep_norm = str(rep).strip().upper()
    outfile = os.path.join(structdir, rep_norm + '.pdb')
    
    # Check if file already exists
    if os.path.exists(outfile):
        if verbose:
            print(f"File already exists: {outfile}")
        return outfile
    
    try:
        # Try current and legacy AFDB model versions (v6 currently most common)
        for url in _candidate_urls(rep_norm):
            result = subprocess.run(
                ['wget', '-q', '-O', outfile, url],
                timeout=timeout,
                capture_output=True,
                text=True
            )

            if result.returncode == 0 and os.path.exists(outfile) and os.path.getsize(outfile) > 0:
                if verbose:
                    print(f"Downloaded: {outfile} ({url})")
                return outfile

            # Clean up partial/failed file before next candidate
            if os.path.exists(outfile):
                os.remove(outfile)

            if verbose and result.stderr:
                print(f"wget error for {rep_norm} @ {url}: {result.stderr[:100]}")

        return None
            
    except subprocess.TimeoutExpired:
        print(f"Timeout downloading {rep}")
        if os.path.exists(outfile):
            os.remove(outfile)
        return None
    except Exception as e:
        print(f"Error downloading {rep}: {e}")
        if os.path.exists(outfile):
            os.remove(outfile)
        return None


def download_pdb_requests(
    rep: str,
    structdir: str,
    timeout: int = 30,
    verbose: bool = False
) -> Optional[str]:
    """
    Alternative: Download a single PDB file using requests library.
    Use this if wget is not available.
    
    Args:
        rep: Protein identifier
        structdir: Directory to save the PDB file
        timeout: Timeout for download in seconds
        verbose: Print debug messages
        
    Returns:
        Path to downloaded file if successful, None otherwise
    """
    try:
        import requests
    except ImportError:
        print("Error: requests library not installed. Install with: pip install requests")
        return None
    
    rep_norm = str(rep).strip().upper()
    outfile = os.path.join(structdir, rep_norm + '.pdb')
    
    if os.path.exists(outfile):
        if verbose:
            print(f"File already exists: {outfile}")
        return outfile
    
    try:
        for url in _candidate_urls(rep_norm):
            response = requests.get(url, timeout=timeout)

            if response.status_code == 200 and response.content:
                with open(outfile, 'wb') as f:
                    f.write(response.content)
                if verbose:
                    print(f"Downloaded: {outfile} ({url})")
                return outfile

            if verbose:
                print(f"HTTP {response.status_code} for {rep_norm} @ {url}")

        return None
            
    except requests.Timeout:
        print(f"Timeout downloading {rep}")
        return None
    except Exception as e:
        print(f"Error downloading {rep}: {e}")
        return None


def download_structures(
    repdf: pd.DataFrame,
    nreps: Optional[int] = None,
    structdir: str = './structs/',
    ncpu: int = 20,
    chunksize: int = 5,
    method: str = 'subprocess',
    timeout: int = 30,
    verbose: bool = False
) -> Tuple[List[str], List[str]]:
    """
    Download multiple PDB files in parallel using multiprocessing.
    
    Args:
        repdf: DataFrame with 'repId' column containing protein identifiers
        nreps: Number of random structures to download (None = all)
        structdir: Directory to save PDB files
        ncpu: Number of parallel processes
        chunksize: Number of items per worker
        method: 'subprocess' (wget) or 'requests'
        timeout: Download timeout in seconds
        verbose: Print debug messages
        
    Returns:
        Tuple of (successful_files, failed_repIds)
        
    Example:
        >>> repdf = pd.read_table('afdbclusters/2-repId_isDark_nMem_repLen_avgLen_repPlddt_avgPlddt_LCAtaxId.tsv')
        >>> successful, failed = download_structures(repdf, nreps=1000, ncpu=20)
        >>> print(f"Downloaded {len(successful)} structures, {len(failed)} failed")
    """
    
    # Create output directory
    if not os.path.exists(structdir):
        os.makedirs(structdir, exist_ok=True)
    
    # Get representative IDs
    reps = repdf.repId.unique()
    if nreps is not None:
        reps = np.random.choice(reps, min(nreps, len(reps)), replace=False)
    
    reps = list(reps)
    print(f"Downloading {len(reps)} structures to {structdir}")
    print(f"Using {method} method with {ncpu} processes, chunksize={chunksize}")
    
    # Select download function
    if method == 'subprocess':
        download_func = download_pdb_subprocess
    elif method == 'requests':
        download_func = download_pdb_requests
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Create partial function with fixed arguments
    import functools
    download_func_partial = functools.partial(
        download_func,
        structdir=structdir,
        timeout=timeout,
        verbose=verbose
    )
    
    # Download in parallel
    successful = []
    failed = []
    
    pool = None
    try:
        with mp.Pool(ncpu) as p:
            pool = p
            for result in tqdm.tqdm(
                p.imap_unordered(download_func_partial, reps, chunksize=chunksize),
                total=len(reps),
                desc="Downloading structures",
                unit="file"
            ):
                if result is not None:
                    successful.append(result)
                else:
                    # Track which rep failed (we don't have direct access to the rep ID here)
                    # This is handled by the print statements in the download function
                    pass
    except KeyboardInterrupt:
        print("\nDownload interrupted by user")
        if pool is not None:
            pool.terminate()
            pool.join()
    
    # Count failed by checking which files weren't downloaded
    failed = [rep for rep in reps if not os.path.exists(os.path.join(structdir, rep + '.pdb'))]
    
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if failed and len(failed) <= 10:
        print(f"Failed IDs: {failed}")
    
    return successful, failed


def verify_downloads(structdir: str, expected_count: Optional[int] = None) -> dict:
    """
    Verify downloaded PDB files.
    
    Args:
        structdir: Directory containing PDB files
        expected_count: Expected number of files (for validation)
        
    Returns:
        Dictionary with statistics
    """
    pdb_files = [f for f in os.listdir(structdir) if f.endswith('.pdb')]
    
    # Check file sizes
    sizes = []
    for pdb_file in pdb_files:
        filepath = os.path.join(structdir, pdb_file)
        size = os.path.getsize(filepath)
        sizes.append(size)
    
    stats = {
        'total_files': len(pdb_files),
        'total_size_mb': sum(sizes) / (1024 * 1024),
        'avg_size_kb': np.mean(sizes) / 1024 if sizes else 0,
        'min_size_kb': np.min(sizes) / 1024 if sizes else 0,
        'max_size_kb': np.max(sizes) / 1024 if sizes else 0,
    }
    
    print("Download Verification")
    print("=" * 60)
    print(f"Total files: {stats['total_files']}")
    print(f"Total size: {stats['total_size_mb']:.1f} MB")
    print(f"Average size: {stats['avg_size_kb']:.1f} KB")
    print(f"Size range: {stats['min_size_kb']:.1f} - {stats['max_size_kb']:.1f} KB")
    
    if expected_count is not None:
        if stats['total_files'] == expected_count:
            print(f"OK: All {expected_count} files present")
        else:
            print(f"ERROR: Expected {expected_count} files, found {stats['total_files']}")
    
    return stats


if __name__ == '__main__':
    # Example usage
    print("FoldTree2 Download Utilities")
    print("Import with: from foldtree2.src.download_utils import download_structures")
