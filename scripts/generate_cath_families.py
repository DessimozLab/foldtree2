#!/usr/bin/env python3
"""Fresh experimental CATH families: one verified PDB chain per NCBI genus."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import copy
import csv
import gzip
import json
import math
from pathlib import Path
import re
import sqlite3
import tempfile
import urllib.error
import urllib.request

from rebuild_cath_families import canonical_taxid, canonical_lineage, digest, download, write_json

SOURCES = {
    'cath-domain-list.txt': 'https://download.cathdb.info/cath/releases/latest-release/cath-classification-data/cath-domain-list.txt',
    'pdb_chain_cath_uniprot.csv.gz': 'https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/csv/pdb_chain_cath_uniprot.csv.gz',
    'pdb_chain_taxonomy.csv.gz': 'https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/csv/pdb_chain_taxonomy.csv.gz',
}
_PLANS = {}


def frozen_plan(path, expected):
    path = Path(path)
    cached = _PLANS.get(str(path))
    frozen = cached[1] if cached else json.loads(path.read_text())
    paths = [path,Path(frozen['taxonomy_db']),*map(Path,frozen['inputs'])]
    signature = tuple((str(p),p.stat().st_mtime_ns,p.stat().st_size) for p in paths)
    if cached and cached[0]==signature and cached[2]==expected:
        return frozen
    if digest(path)!=expected:raise ValueError('Fresh generation plan changed')
    for source,sha in frozen['inputs'].items():
        if digest(source)!=sha:raise ValueError('Curated generation source changed')
    if digest(frozen['taxonomy_db'])!=frozen['taxonomy_sha256']:
        raise ValueError('Generation taxonomy changed')
    _PLANS[str(path)]=(signature,frozen,expected)
    return frozen


def genus_of(db, taxid):
    taxid = canonical_taxid(db, int(taxid))
    visited = set()
    while taxid != 1 and taxid not in visited:
        visited.add(taxid)
        row = db.execute('SELECT parent,rank FROM species WHERE taxid=?', (taxid,)).fetchone()
        if not row:
            break
        if row[1] == 'genus':
            return taxid
        taxid = int(row[0])
    raise ValueError('No classified NCBI genus')


def read_csv(path):
    with gzip.open(path, 'rt', newline='') as stream:
        yield from csv.DictReader(line for line in stream if not line.startswith('#'))


def plan(root, database, max_families=2000, max_genera=100):
    inputs = root/'inputs'; inputs.mkdir()
    hashes = {}
    for name, url in SOURCES.items():
        download(url, inputs/name)
        hashes[str((inputs/name).resolve())] = digest(inputs/name)
    domains = {}
    for line in (inputs/'cath-domain-list.txt').read_text().splitlines():
        if line.startswith('#') or not line.strip():
            continue
        fields = line.split()
        family = '.'.join(fields[1:5])
        if not re.fullmatch(r'\d+(?:\.\d+){3}', family):
            raise ValueError('Unexpected CATH family identifier')
        resolution = float(fields[-1])
        domains[fields[0]] = (family, resolution if resolution > 0 else math.inf)
    taxa = defaultdict(set)
    for row in read_csv(inputs/'pdb_chain_taxonomy.csv.gz'):
        taxa[(row['PDB'].lower(),row['CHAIN'])].add(row['TAX_ID'])
    mappings = list(read_csv(inputs/'pdb_chain_cath_uniprot.csv.gz'))
    accessions = defaultdict(set)
    for row in mappings:
        accessions[(row['PDB'].lower(),row['CHAIN'])].add(row['SP_PRIMARY'])
    groups = defaultdict(lambda: defaultdict(dict)); exclusions = Counter()
    with sqlite3.connect(f'file:{database}?mode=ro',uri=True) as db:
        taxcache = {}
        for row in mappings:
            pdb,chain,accession = row['PDB'].lower(),row['CHAIN'],row['SP_PRIMARY']
            key = (pdb,chain)
            if row['CATH_ID'] not in domains:
                exclusions['CATH_domain_not_in_snapshot'] += 1; continue
            if len(accessions[key]) != 1 or len(taxa[key]) != 1:
                exclusions['ambiguous_chain_accession_or_taxonomy'] += 1; continue
            if not re.fullmatch(r'[a-z0-9]{4}',pdb) or not re.fullmatch(r'[A-Za-z0-9]{6,10}(?:-\d+)?',accession):
                exclusions['unsafe_identifiers'] += 1; continue
            taxid = next(iter(taxa[key]))
            if taxid not in taxcache:
                try:
                    canonical = canonical_taxid(db,int(taxid)); genus = genus_of(db,canonical)
                    parent = db.execute('SELECT parent FROM species WHERE taxid=?',(canonical,)).fetchone()[0]
                    taxcache[taxid] = (canonical,genus,canonical_lineage(db,parent))
                except ValueError:
                    taxcache[taxid] = None
            if taxcache[taxid] is None:
                exclusions['unresolved_taxonomy_or_genus'] += 1; continue
            canonical,genus,lineage = taxcache[taxid]
            family,resolution = domains[row['CATH_ID']]
            candidate = {'pdb':pdb,'chain':chain,'accession':accession,'domain':row['CATH_ID'],
                'taxid':canonical,'genus':genus,'lineage':lineage,
                'resolution':None if math.isinf(resolution) else resolution}
            groups[family][genus][(pdb,chain,accession)] = candidate
    families = []
    for family, genera in sorted(groups.items()):
        if len(genera) < 4:
            exclusions['families_fewer_than_four_genera'] += 1; continue
        selected = {}
        # Stable ordering; no random or completion-order-dependent representative selection.
        for genus,candidates in sorted(genera.items())[:max_genera]:
            ordered = sorted(candidates.values(),key=lambda c:(c['resolution'] or math.inf,c['pdb'],c['chain'],c['accession']))
            selected[str(genus)] = ordered[:3]
            exclusions['within_genus_alternative_chains'] += max(0,len(ordered)-1)
        families.append({'family':family,'genera_available':len(genera),'genera':selected})
    payload = {'schema':'cath_genus_generation_plan_v1','families':families[:max_families],
        'eligible_families_before_cap':len(families),'max_families':max_families,'max_genera':max_genera,
        'source_urls':SOURCES,'inputs':hashes,'taxonomy_db':str(database),'taxonomy_sha256':digest(database),
        'exclusions':dict(exclusions),'selection':'one chain per canonical genus; up to three ordered alternatives',
        'structure_scope':'full observed experimental chain, matching legacy CATH baseline; not domain cropping'}
    write_json(root/'plan.json',payload)
    write_json(root/'plan_summary.json',{k:v for k,v in payload.items() if k!='families'})
    return payload


def experimental_chain(path, candidate):
    from Bio.PDB import MMCIFParser
    from Bio.SeqUtils import seq1
    structure = MMCIFParser(QUIET=True).get_structure(candidate['pdb'],str(path))
    model = next(structure.get_models())  # Explicitly use the first deposited model.
    if candidate['chain'] not in model:
        raise ValueError('Mapped author chain is absent from deposited mmCIF')
    chain = copy.deepcopy(model[candidate['chain']])
    # Keep observed standard protein residues with CA, as used by pdb_sequence.
    for residue in list(chain):
        if residue.id[0] != ' ' or 'CA' not in residue:
            chain.detach_child(residue.id)
    residues = list(chain)
    sequence = ''.join(seq1(r.resname) for r in residues)
    if len(sequence) < 20 or 'X' in sequence:
        raise ValueError('Too few/unknown observed experimental residues')
    chain.detach_parent()
    chain.id = 'A'  # PDB-format single-chain alias only; no coordinate/residue repair.
    return chain,sequence


def cached_structure(root, pdb):
    import fcntl
    cache = root/'structures'; cache.mkdir(exist_ok=True)
    path = cache/(pdb+'.cif')
    with (cache/(pdb+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not path.exists():
            # Bound this dataset's footprint even if some deposited entries are enormous.
            limit=50*1024**2;budget=50*1024**3
            accounting=cache/'storage.json'
            with (cache/'storage.lock').open('a') as storage:
                fcntl.flock(storage,fcntl.LOCK_EX)
                used=json.loads(accounting.read_text())['reserved_or_used_bytes'] if accounting.exists() else 0
                if used+limit>budget:raise RuntimeError('50 GiB structure cache budget reached')
                write_json(accounting,{'reserved_or_used_bytes':used+limit})
            temporary=None;size=0
            try:
                request=urllib.request.Request(f'https://files.rcsb.org/download/{pdb.upper()}.cif',
                    headers={'User-Agent':'FoldTree2-CATH-genus/1'})
                with urllib.request.urlopen(request,timeout=60) as response, tempfile.NamedTemporaryFile(
                        dir=cache,delete=False) as stream:
                    temporary=Path(stream.name)
                    for block in iter(lambda:response.read(1024**2),b''):
                        size+=len(block)
                        if size>limit:raise ValueError('Deposited mmCIF exceeds 50 MiB cap')
                        stream.write(block)
                temporary.replace(path)
            finally:
                if temporary is not None and temporary.exists():temporary.unlink()
                with (cache/'storage.lock').open('a') as storage:
                    fcntl.flock(storage,fcntl.LOCK_EX)
                    used=json.loads(accounting.read_text())['reserved_or_used_bytes']
                    write_json(accounting,{'reserved_or_used_bytes':used-limit+(size if path.exists() else 0)})
    return path


def generate_family(task):
    root,family = task; root = Path(root)
    directory = root/'CATH_data'/family['family']
    proteins,failures,rows = {},[],[]
    from Bio.PDB import PDBIO
    from Bio.PDB.PDBExceptions import PDBException
    for genus,candidates in family['genera'].items():
        for candidate in candidates:
            accession = candidate['accession']
            if accession in proteins:
                failures.append({'candidate':candidate,'reason':'accession already selected'});continue
            try:
                path = cached_structure(root,candidate['pdb'])
                chain,sequence = experimental_chain(path,candidate)
                directory.mkdir(parents=True,exist_ok=True); (directory/'structs').mkdir(exist_ok=True)
                output = directory/'structs'/(accession+'.pdb')
                writer=PDBIO(); writer.set_structure(chain); writer.save(str(output))
                from curnagl_tcs import pdb_sequence
                if pdb_sequence(output) != sequence:
                    raise ValueError('Exported PDB differs from observed deposited sequence')
                proteins[accession] = {'candidate':candidate,'source_cif':str(path.resolve()),
                    'source_cif_sha256':digest(path),'pdb_sha256':digest(output),'observed_sequence':sequence}
                rows.append({'query':accession,'Entry':accession,'Sequence':sequence,
                    'Sequence_source':'observed_pdb_residues','Taxonomic lineage (Ids)':candidate['lineage'],
                    'Organism ID':candidate['taxid'],'Genus ID':genus,'PDB':candidate['pdb'],
                    'Author chain':candidate['chain'],'CATH domain':candidate['domain']})
                break
            except (ValueError, KeyError, PDBException, urllib.error.HTTPError) as error:
                if isinstance(error, urllib.error.HTTPError) and error.code not in {404, 410}:
                    raise
                failures.append({'candidate':candidate,'reason':str(error)})
                # Do not let a failed export masquerade as a selected chain on retry.
                if directory.exists():
                    output=directory/'structs'/(accession+'.pdb')
                    if output.exists():output.rename(output.with_suffix('.failed-pdb'))
    result={'family':family['family'],'retained':len(rows),'available_genera':family['genera_available'],
            'candidate_failures':failures,'status':'passed' if len(rows)>=4 else 'fewer_than_four_successful_genera'}
    if len(rows)>=4:
        for filename in ('sequence_dataset.csv','finalset.csv'):
            with (directory/filename).open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        (directory/'identifiers.txt').write_text('\n'.join(proteins)+'\n')
        (directory/'sequences.fasta').write_text(''.join(f'>{r["query"]}\n{r["Sequence"]}\n' for r in rows))
        write_json(directory/'rebuild_provenance.json',{'schema':'cath_generated_genus_v1','family':family['family'],
            'plan':str((root/'plan.json').resolve()),'plan_sha256':digest(root/'plan.json'),
            'metadata_sha256':digest(directory/'finalset.csv'),'proteins':proteins,
            'missing_residues_filled':False,'inferred_genus_labels':False})
        validate_generated(directory)
    elif directory.exists():
        # Preserve unsuccessful inputs/reports but keep them outside the eligible cohort root.
        rejected=root/'rejected';rejected.mkdir(exist_ok=True);directory.rename(rejected/directory.name)
    write_json(root/'reports'/(family['family']+'.json'),result)
    print(json.dumps({k:v for k,v in result.items() if k!='candidate_failures'}),flush=True)
    return result


def validate_generated(directory):
    directory=Path(directory)
    proof=json.loads((directory/'rebuild_provenance.json').read_text())
    if proof['schema']!='cath_generated_genus_v1':raise ValueError('Unknown fresh-generation schema')
    frozen=frozen_plan(proof['plan'],proof['plan_sha256'])
    planned=next(f for f in frozen['families'] if f['family']==proof['family'])
    proteins=proof['proteins'];genera=[]
    if len(proteins)<4 or {p.stem for p in (directory/'structs').glob('*.pdb')}!=set(proteins):
        raise ValueError('Incomplete fresh family cohort')
    if digest(directory/'finalset.csv')!=proof['metadata_sha256']:
        raise ValueError('Fresh metadata changed')
    with sqlite3.connect(f'file:{frozen["taxonomy_db"]}?mode=ro',uri=True) as db:
        for accession,record in proteins.items():
            candidate=record['candidate'];genus=genus_of(db,candidate['taxid']);genera.append(genus)
            if genus!=candidate['genus'] or candidate not in planned['genera'].get(str(genus),[]):
                raise ValueError('Chain not in curated genus-selection plan')
            if accession!=candidate['accession'] or digest(record['source_cif'])!=record['source_cif_sha256']:
                raise ValueError('Deposited experimental source changed')
            if digest(directory/'structs'/(accession+'.pdb'))!=record['pdb_sha256']:
                raise ValueError('Generated experimental chain changed')
            _,observed=experimental_chain(Path(record['source_cif']),candidate)
            if observed!=record['observed_sequence']:
                raise ValueError('Observed source sequence changed')
    if len(genera)!=len(set(genera)):
        raise ValueError('Duplicate genus in generated family')
    return proof,frozen


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--taxonomy-db',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--max-families',type=int,default=2000)
    parser.add_argument('--max-genera',type=int,default=100)
    parser.add_argument('--plan-only',action='store_true')
    parser.add_argument('--from-plan',action='store_true')
    parser.add_argument('--limit-families',type=int,help='Generate only the first N planned families for a separate pilot')
    args=parser.parse_args();root=args.out.resolve()
    if args.workers<1 or args.max_families<1 or args.max_genera<4:
        parser.error('Invalid resource/cohort limits')
    database=args.taxonomy_db.resolve()
    if digest(database)!=json.loads(database.with_name('snapshot.json').read_text())['sqlite_sha256']:
        raise ValueError('Existing taxonomy snapshot changed')
    if args.from_plan:
        frozen=json.loads((root/'plan.json').read_text())
        if (root/'CATH_data').exists():raise FileExistsError('Generation already started; use a new output')
        if frozen['taxonomy_sha256']!=digest(database):raise ValueError('Taxonomy differs from plan')
    else:
        if root.exists():raise FileExistsError('Choose a new generation directory')
        root.mkdir(parents=True);frozen=plan(root,database,args.max_families,args.max_genera)
    if args.plan_only:return
    frozen_plan(root/'plan.json',digest(root/'plan.json'))
    (root/'CATH_data').mkdir()
    selected=frozen['families'][:args.limit_families] if args.limit_families else frozen['families']
    tasks=[(str(root),family) for family in selected]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:results=list(pool.map(generate_family,tasks))
    write_json(root/'summary.json',{'requested_families':len(results),
        'passed_families':sum(r['status']=='passed' for r in results),
        'retained_protein_tips':sum(r['retained'] for r in results if r['status']=='passed'),
        'genus_duplicates_allowed':False,'plan_sha256':digest(root/'plan.json'),'results':results})
    if not any(r['status']=='passed' for r in results):
        raise ValueError('No fresh families passed generation checks')


if __name__=='__main__':main()
