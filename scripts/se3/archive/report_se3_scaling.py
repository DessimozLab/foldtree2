"""Report a matched 20-warmup/100-update scaling window from persistent events."""
import argparse
import json
from pathlib import Path
import statistics


def report(path):
    rows=[json.loads(x) for x in Path(path).read_text().splitlines()]
    updates=[r for r in rows if r['event']=='update']
    timed=updates[20:120]
    if len(timed)!=100:raise ValueError('Need 20 warmup and 100 timed optimizer updates')
    ready=next(r for r in rows if r['event']=='ready')
    seconds=sum(r['update_seconds'] for r in timed)
    return {'world_size':ready['topology']['world_size'],'updates':100,
            'structures_per_second':1600/seconds,'seconds':seconds,
            'peak_gpu_bytes':max(r['peak_memory_bytes'] for r in timed),
            'mean_last_micro_data_wait_seconds':statistics.mean(r['data_wait_seconds'] for r in timed),
            'bundle':ready['bundle']}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('events',nargs='+');p.add_argument('--output',required=True)
    a=p.parse_args();reports=[report(x) for x in a.events]
    baseline=reports[0]
    for r in reports:
        if r['bundle']!=baseline['bundle']:raise ValueError('Scaling cohort differs')
        r['efficiency']=r['structures_per_second']/baseline['structures_per_second']/(r['world_size']/baseline['world_size'])
    Path(a.output).write_text(json.dumps(reports,indent=2)+'\n');print(json.dumps(reports))
