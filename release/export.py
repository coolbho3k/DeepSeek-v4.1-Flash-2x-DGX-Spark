# SPDX-License-Identifier: AGPL-3.0-only
"""Export only public recipe files to a NEW directory; preserve the campaign."""
import argparse
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
FILES=('README.md','LICENSE','CREDITS.md','THIRD_PARTY_NOTICES.md','.gitignore',
       '.gitattributes','.env.ds41.example','start-server.sh','stop-server.sh','recipe-lock.json',
       'docs/display-memory.md','docs/configuration.md','docs/release-validation.md',
       'docs/release-maintenance.md','docs/kernel-batch-performance.md',
       'docs/repository-review.md',
       'probes/compare_kernel_batch.py')
TREES=('release','tests','.github')


def export(destination):
    destination=Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise ValueError('Choose a new destination; existing files are never replaced')
    if destination==ROOT or ROOT.is_relative_to(destination) or destination.is_relative_to(ROOT):
        raise ValueError('Export outside the campaign checkout to avoid recursive copies')
    paths=[ROOT/name for name in FILES]
    for tree in TREES:
        paths.extend(p for p in (ROOT/tree).rglob('*') if p.is_file()
                     and '__pycache__' not in p.parts and not p.name.endswith(('.pyc','.local.json')))
    if any(p.is_symlink() for p in paths):raise ValueError('No symlinks in public export')
    destination.mkdir(parents=True)
    for source in paths:
        target=destination/source.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
    print(f'Exported {len(paths)} files to {destination}; no server operations performed.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    export(p.parse_args().output)
