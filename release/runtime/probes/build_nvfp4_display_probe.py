# SPDX-License-Identifier: AGPL-3.0-only
"""Build a 4-MiB benchmark-only variant of the unchanged display allocator."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-directory', type=Path, required=True)
    parser.add_argument('--cuda', type=Path, default=Path('/usr/local/cuda'))
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1] / 'sources/display_kv.c'
    raw = source.read_text()
    anchor = 'display!=1792UL*1024*1024'
    assert raw.count(anchor) == 1
    # Only this independent test allocation changes size. The serving
    # allocator, its binary and its existing live pool are never modified.
    candidate = raw.replace(anchor, 'display!=4UL*1024*1024')
    candidate = candidate.replace('and1.75GiB display', 'and4MiB display (benchmark only)')
    out = args.output_directory.resolve()
    out.mkdir(parents=True, exist_ok=False)
    src = out / 'display_probe.c'
    src.write_text(candidate)
    library = out / 'display_probe.so'
    subprocess.run(['gcc', '-O2', '-fPIC', '-shared', '-I' + str(args.cuda / 'include'),
        '-I/usr/include/libdrm', str(src), '-L' + str(args.cuda / 'lib64/stubs'),
        '-lcuda', '-o', str(library)], check=True)
    report = dict(display_bytes=4*2**20, ordinary_bytes=0,
        production_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        probe_source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
        probe_library_sha256=hashlib.sha256(library.read_bytes()).hexdigest())
    (out / 'build.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
