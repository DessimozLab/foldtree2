import io
import tarfile
import pytest
from scripts.verify_se3_release import REQUIRED, verify


def archive(path, omit=None):
    with tarfile.open(path, 'w:gz') as output:
        for name in REQUIRED - ({omit} if omit else set()):
            payload = (b'gotennet-pytorch==0.3.1\npytorch-lightning==2.6.0\n'
                       if name.endswith('requirements.txt') else b'fixture\n')
            member = tarfile.TarInfo(name); member.size = len(payload)
            output.addfile(member, io.BytesIO(payload))


def test_release_contains_the_dependency_file(tmp_path):
    path = tmp_path/'complete.tar.gz'; archive(path)
    assert verify(path)['verified']


def test_release_rejects_missing_ignored_requirements(tmp_path):
    path = tmp_path/'incomplete.tar.gz'; archive(path, 'alps/se3/requirements.txt')
    with pytest.raises(ValueError, match='requirements.txt'):
        verify(path)
