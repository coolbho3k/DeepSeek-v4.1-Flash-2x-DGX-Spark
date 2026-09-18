# SPDX-License-Identifier: AGPL-3.0-only
"""Prepare an UNQUALIFIED local candidate; never deploy, test, or start it.

Standard library only. No model access, network, subprocess, credential reads,
or active-profile writes. The parent and fresh destination must be explicit.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat

SOURCE = Path(__file__).resolve().parent
MODULES = ('__init__', 'policy', 'transport', 'attention', 'packed', 'integration')
MAX_FILE, MAX_TOTAL = 8 * 2**20, 64 * 2**20


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()


def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key')
        result[key] = value
    return result


def safe_name(name):
    if (not isinstance(name, str) or not name or '\\' in name or '\0' in name
            or PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts
            or str(PurePosixPath(name)) != name or name == '.'):
        raise ValueError('Unsafe bundle path')
    return name


def bounded_read(path):
    before = path.lstat()
    if (path.resolve() != path or not stat.S_ISREG(before.st_mode)
            or before.st_size > MAX_FILE):
        raise ValueError('Expected bounded, unredirected regular file: ' + str(path))
    raw = path.read_bytes()
    after = path.lstat()
    fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    if any(getattr(before, k) != getattr(after, k) for k in fields):
        raise ValueError('Source changed while reading')
    return raw


def load_parent(parent, expected_sha):
    parent = Path(parent).absolute()
    if parent.resolve() != parent or not parent.is_dir():
        raise ValueError('Use an unredirected parent directory')
    raw = bounded_read(parent / 'bundle-manifest.json')
    if not re.fullmatch('[0-9a-f]{64}', expected_sha) or sha(raw) != expected_sha:
        raise ValueError('Parent manifest differs from the separately supplied digest')
    manifest = json.loads(raw, object_pairs_hook=unique)
    if (manifest.get('format') not in tuple('ds41_runtime_inputs_v' + str(n) for n in range(1, 6))
            or any(manifest.get(k) is not False for k in
                   ('standalone_runtime', 'clean_rebuild_qualified', 'publication_approved'))):
        raise ValueError('Unexpected parent format or qualification claim')
    files = manifest['files']
    if not isinstance(files, dict) or not 1 <= len(files) <= 1000:
        raise ValueError('Invalid parent inventory')
    actual = set()
    for path in parent.rglob('*'):
        info = path.lstat()
        if path.resolve() != path:
            raise ValueError('Redirected bundle entry')
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode):
            raise ValueError('Special file in bundle')
        actual.add(path.relative_to(parent).as_posix())
    if actual != set(files) | {'bundle-manifest.json'}:
        raise ValueError('Unexpected or missing parent files')
    payload, total = {}, 0
    for name, row in files.items():
        safe_name(name)
        if (name == 'bundle-manifest.json' or not isinstance(row, dict)
                or type(row.get('bytes')) is not int or not 0 <= row['bytes'] <= MAX_FILE
                or not isinstance(row.get('sha256'), str)
                or not re.fullmatch('[0-9a-f]{64}', row['sha256'])):
            raise ValueError('Invalid parent file descriptor')
        total += row['bytes']
        if total > MAX_TOTAL:
            raise ValueError('Parent exceeds small-runtime-input scope')
        data = bounded_read(parent / name)
        if len(data) != row['bytes'] or sha(data) != row['sha256']:
            raise ValueError('Parent file changed: ' + name)
        payload[name] = data
    return manifest, payload


def transform(parent_payload, mode):
    """Pure CPU transformation; never changes parent_payload or its files."""
    if mode not in ('off', 'query', 'balanced', 'concurrent'):
        raise ValueError('Unknown overlap mode')
    payload = dict(parent_payload)
    if any('/dcp_overlap/' in name for name in payload):
        raise ValueError('Already an overlap candidate; use the original parent')
    history = {name: {sha(raw)} for name, raw in payload.items() if name.endswith('.py')}

    def edit(name, old, new):
        source = payload[name].decode()
        if source.count(old) != 1:
            raise ValueError('Changed source anchor in ' + name + ': ' + old[:100])
        payload[name] = source.replace(old, new).encode()

    for name in MODULES:
        payload['serving/ds41/dcp_overlap/' + name + '.py'] = bounded_read(SOURCE / (name + '.py'))
    edit('serving/ds41/dcp_overlap/policy.py', "MODE = 'off'", 'MODE = ' + repr(mode))
    edit('serving/ds41/vllm_fp4_main.py', '        workspace.register()\n',
         '        from .dcp_overlap.integration import wrap_forward\n'
         '        forward = wrap_forward(forward)\n'
         '        workspace.register()\n')
    name = 'serving/spark_dcp_communication.py'
    edit(name, 'import hashlib\n',
         'import hashlib\nfrom ds41.dcp_overlap.integration import forward_admitted as _ds41_overlap_forward_admitted\n')
    # Keep the quoted anchor: combined_miaai subsequently adds rank to it.
    text = "'all_packed = group.all_gather(_ds41_pack_result(partial, lse), dim=0)'"
    edit(name, text + ' not in forward.__ds41_patch_source__',
         'not _ds41_overlap_forward_admitted(forward, ' + text + ')')
    edit('serving/combined_worker.py', '    init_device = _init_device\n',
         '    from ds41.dcp_overlap.integration import wrap_init_device as _wrap_overlap_init\n'
         '    init_device = _wrap_overlap_init(_init_device)\n')
    edit('serving/spark_sparse_slots.py', '    forward.__globals__[KEY]=candidate\n',
         '    from ds41.dcp_overlap.integration import bind_sparse_mapper\n'
         '    bind_sparse_mapper(forward, candidate)\n')
    edit('serving/spark_combined_miaai.py', 'DESCRIPTOR = dict(kernel_batch=KERNEL_BATCH,',
         'DCP_OVERLAP = ' + repr(dict(mode=mode, experimental=True, gpu_qualified=False,
             tensor_parallel_size=2, decode_context_parallel_size=2,
             quantization_unchanged=True, memory_limits_unchanged=True)) + '\n'
         'DESCRIPTOR = dict(dcp_overlap=DCP_OVERLAP, kernel_batch=KERNEL_BATCH,')

    name = 'serving/spark_backend_attestation.py'
    source = payload[name].decode()
    assignments = [n for n in ast.parse(source).body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'PRIVATE_SOURCES' for t in n.targets)]
    if len(assignments) != 1:
        raise ValueError('Changed attestation inventory')
    assignment = assignments[0]
    pins = ast.literal_eval(assignment.value)
    for module in MODULES:
        relative = 'ds41/dcp_overlap/' + module + '.py'
        pins[relative] = sha(payload['serving/' + relative])
    edit(name, ast.get_source_segment(source, assignment), 'PRIVATE_SOURCES = ' + repr(pins))

    # Propagate affected source pins, including worker/startup guards.
    # A hash cycle fails instead of weakening an attestation check.
    final_updates = {}
    for _ in range(32):
        for name, values in history.items():
            values.add(sha(payload[name]))
        updates = {old: sha(payload[name]) for name, values in history.items()
                   for old in values if old != sha(payload[name])}
        final_updates.update(updates)
        changed = False
        for name, raw in list(payload.items()):
            if not name.endswith('.py') or not name.startswith(('serving/', 'tools/')):
                continue
            for old, new in updates.items():
                raw = raw.replace(old.encode(), new.encode())
            if raw != payload[name]:
                payload[name], changed = raw, True
        if not changed:
            break
    else:
        raise ValueError('Source pins failed to converge')

    requirements = json.loads(payload['runtime-requirements.json'], object_pairs_hook=unique)
    def refresh(value):
        if isinstance(value, dict):
            return {k: refresh(v) for k, v in value.items()}
        if isinstance(value, list):
            return [refresh(v) for v in value]
        return final_updates.get(value, value) if isinstance(value, str) else value
    requirements = refresh(requirements)
    requirements['dcp_overlap_candidate'] = dict(mode=mode, status='unqualified',
        cpu_tests_run=False, gpu_tests_run=False, serving_tests_run=False,
        publication_approved=False, inherited_qualification_applies_to_parent_only=True,
        serving_settings_unchanged=True, new_persistent_gpu_tensor_workspace_bytes=0)
    requirements['loaded_backend_verification']['sha256'] = sha(payload['serving/spark_backend_attestation.py'])
    payload['runtime-requirements.json'] = encoded(requirements)
    payload['serving/overlay-manifest.json'] = encoded({name.removeprefix('serving/'): sha(raw)
        for name, raw in payload.items() if name.startswith('serving/') and name != 'serving/overlay-manifest.json'})
    for name in (*[module + '.py' for module in MODULES],
                 'README.md', 'probe_gpu.py', 'test_cpu.py', 'prepare.py', 'run_pair.py', 'summarize.py'):
        payload['experiments/dcp_overlap/' + name] = bounded_read(SOURCE / name)
    for name, raw in payload.items():
        if name.endswith('.py'):
            compile(raw, name, 'exec')  # Future packaging check, not executed on import.
    return payload


def prepare(parent, expected_sha, output, mode):
    parent, output = Path(parent).absolute(), Path(output).absolute()
    if (output.resolve() != output or not output.parent.is_dir() or output.exists()
            or output.is_relative_to(parent) or parent.is_relative_to(output)):
        raise ValueError('Use a fresh unredirected sibling directory, not an existing bundle')
    previous, source = load_parent(parent, expected_sha)
    payload = transform(source, mode)
    if (len(payload) > 1000 or sum(map(len, payload.values())) > MAX_TOTAL
            or any(len(raw) > MAX_FILE for raw in payload.values())):
        raise ValueError('Candidate exceeds bounded runtime-input scope')
    manifest = dict(format=previous['format'], standalone_runtime=False,
        clean_rebuild_qualified=False, publication_approved=False, serving_qualified=False,
        variant='experimental_dcp_overlap_' + mode, parent_manifest_sha256=expected_sha,
        files={n: dict(bytes=len(raw), sha256=sha(raw)) for n, raw in sorted(payload.items())})
    output.mkdir(mode=0o700)
    for name, raw in payload.items():
        path = output / safe_name(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as handle:
            handle.write(raw)
    raw = encoded(manifest)
    with (output / 'bundle-manifest.json').open('xb') as handle:
        handle.write(raw)
    load_parent(output, sha(raw))
    return dict(candidate=str(output), manifest_sha256=sha(raw), mode=mode,
                deployed=False, tested=False, public_release_changed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--parent-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('off', 'query', 'balanced', 'concurrent'), required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.parent, args.parent_sha256, args.output, args.mode), indent=2))


if __name__ == '__main__':
    main()
