"""Resolve the CSCS mirror tag to an immutable registry digest before allocation."""
import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen


def pin(edf, output):
    path=Path(edf);text=path.read_text()
    tag='jfrog.svc.cscs.ch/docker-group-csstaff/alps-images/ngc-pytorch:26.02-py3-alps6'
    if 'image = "'+tag+'"' not in text:raise ValueError('Expected reviewed CSCS image tag')
    url='https://jfrog.svc.cscs.ch/v2/docker-group-csstaff/alps-images/ngc-pytorch/manifests/26.02-py3-alps6'
    accept=','.join(['application/vnd.oci.image.index.v1+json','application/vnd.docker.distribution.manifest.list.v2+json',
                     'application/vnd.oci.image.manifest.v1+json','application/vnd.docker.distribution.manifest.v2+json'])
    with urlopen(Request(url,headers={'Accept':accept}),timeout=30) as response:
        digest=response.headers['Docker-Content-Digest'];manifest=json.load(response)
    if not digest or not digest.startswith('sha256:'):raise ValueError('Registry did not provide content digest')
    image=tag.rsplit(':',1)[0]+'@'+digest
    record={'tag':tag,'digest':digest,'image':image,'manifest':manifest}
    Path(output).write_text(json.dumps(record,indent=2)+'\n')
    target=path.with_name('extended-pinned.toml')
    target.write_text(text.replace(tag,image));print(json.dumps({'edf':str(target),'image':image}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--edf',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();pin(a.edf,a.output)
