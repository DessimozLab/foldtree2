"""Resolve the CSCS mirror tag to an immutable registry digest before allocation."""

import os as _bootstrap_os, sys as _bootstrap_sys
_bootstrap_root = _bootstrap_os.path.dirname(_bootstrap_os.path.dirname(_bootstrap_os.path.dirname(_bootstrap_os.path.abspath(__file__))))
if _bootstrap_root not in _bootstrap_sys.path:
    _bootstrap_sys.path.insert(0, _bootstrap_root)
import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import urlencode


def pin(edf, output, registry='ghcr'):
    path=Path(edf);text=path.read_text()
    tag='jfrog.svc.cscs.ch/docker-group-csstaff/alps-images/ngc-pytorch:26.02-py3-alps6'
    if 'image = "'+tag+'"' not in text:raise ValueError('Expected reviewed CSCS image tag')
    headers = {}
    if registry == 'ghcr':
        tag = 'ghcr.io/eth-cscs/alps-extended-images/ngc-pytorch:26.02-py3-alps6'
        scope = 'repository:eth-cscs/alps-extended-images/ngc-pytorch:pull'
        # Anonymous public-registry pull token stays in memory and is never logged.
        with urlopen('https://ghcr.io/token?' + urlencode({'service':'ghcr.io','scope':scope}), timeout=30) as response:
            token = json.load(response)['token']
        headers['Authorization'] = 'Bearer ' + token
        url = 'https://ghcr.io/v2/eth-cscs/alps-extended-images/ngc-pytorch/manifests/26.02-py3-alps6'
    else:
        url='https://jfrog.svc.cscs.ch/v2/docker-group-csstaff/alps-images/ngc-pytorch/manifests/26.02-py3-alps6'
    accept=','.join(['application/vnd.oci.image.index.v1+json','application/vnd.docker.distribution.manifest.list.v2+json',
                     'application/vnd.oci.image.manifest.v1+json','application/vnd.docker.distribution.manifest.v2+json'])
    headers['Accept'] = accept
    with urlopen(Request(url,headers=headers),timeout=30) as response:
        digest=response.headers['Docker-Content-Digest'];manifest=json.load(response)
    if not digest or not digest.startswith('sha256:'):raise ValueError('Registry did not provide content digest')
    image=tag.rsplit(':',1)[0]+'@'+digest
    architecture = None
    if manifest.get('manifests'):
        platforms = [m.get('platform', {}).get('architecture') for m in manifest['manifests']]
        if 'arm64' not in platforms:
            raise ValueError('Image index does not contain ARM64')
        architecture = 'arm64'
    else:
        blob = url.split('/manifests/')[0] + '/blobs/' + manifest['config']['digest']
        with urlopen(Request(blob, headers=headers), timeout=30) as response:
            architecture = json.load(response)['architecture']
        if architecture != 'arm64':
            raise ValueError('Expected ARM64 image')
    record={'tag':tag,'digest':digest,'image':image,'architecture':architecture,'manifest':manifest}
    Path(output).write_text(json.dumps(record,indent=2)+'\n')
    target=path.with_name('extended-pinned.toml')
    target.write_text(text.replace('jfrog.svc.cscs.ch/docker-group-csstaff/alps-images/ngc-pytorch:26.02-py3-alps6', image));print(json.dumps({'edf':str(target),'image':image}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--edf',required=True);p.add_argument('--output',required=True);p.add_argument('--registry',choices=['ghcr','jfrog'],default='ghcr')
    a=p.parse_args();pin(a.edf,a.output,a.registry)
