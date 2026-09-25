# SPDX-License-Identifier: AGPL-3.0-only
"""Rebuild one SciCode step test script and report its exit code and peak RSS.

Usage: peak_step.py <step> <generated_code_dir> <split>
"""
import json, resource, subprocess, sys
from pathlib import Path
from datasets import load_dataset
E_ = '/home/emi/code/ds41/artifacts/evals/scicode'
step, code_dir = sys.argv[1], sys.argv[2]
prob = step.split('.')[0]
ds = load_dataset('SciCode1/SciCode', split=sys.argv[3])
sub = next(s for r in ds if r['problem_id'] == prob for s in r['sub_steps'] if s['step_number'] == step)
tests = sub['test_cases']
src = Path(code_dir, f'{step}.py').read_text()
src += "\nfrom scicode.parse.parse import process_hdf5_to_tuple\n"
src += f"targets = process_hdf5_to_tuple('{step}', {len(tests)}, '{E_}/eval/data/gdrive/test_data.h5')\n"
for i, t in enumerate(tests):
    src += f"target = targets[{i}]\n" + t + "\n"
Path('/tmp/step_test.py').write_text(src)
r = subprocess.run(['python', '/tmp/step_test.py'], capture_output=True, text=True)
print(json.dumps(dict(step=step, returncode=r.returncode, peak_rss_mib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss // 1024, stderr=r.stderr[-300:])))
