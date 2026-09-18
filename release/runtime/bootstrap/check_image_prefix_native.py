"""CPU-only actual-image import and spawn checks for the reviewed entrypoint.

Run only in an isolated no-GPU container with a staged /opt/ds41-serving
overlay and a dedicated /output mount. No model files are needed or loaded.
"""
import argparse
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace as NS

sys.dont_write_bytecode = True
sys.path.insert(0, '/opt/ds41-serving')
if Path('/dev/nvidia0').exists() or Path('/dev/nvidiactl').exists():
    raise RuntimeError('Native CPU qualification must not have GPU devices')
# Exercise the REAL serving entrypoint's out-of-main bootstrap in both the
# parent and every spawn interpreter, without entering its server CLI.
runpy.run_path('/opt/ds41-serving/serve.py', run_name='__ds41_cpu_entrypoint__')


def check():
    import torch
    from ds41 import vllm_vision_inputs as vision
    from spark_image_prefix import install, OVERRIDE_SHA
    from vllm.v1.core.kv_cache_manager import KVCacheManager, KVCacheBlocks
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.request import Request
    from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
    from vllm import SamplingParams
    assert not torch.cuda.is_initialized()
    assert install() is vision
    assert hashlib.sha256(Path(vision.__file__).read_bytes()).hexdigest() == OVERRIDE_SHA
    vision.register()
    installed = KVCacheManager.get_computed_blocks
    assert installed is vision.PATCHED_PREFIX_LOOKUP
    assert Scheduler._try_schedule_encoder_inputs is vision.schedule_whole_images
    vision.register()
    assert KVCacheManager.get_computed_blocks is installed
    records = []
    for label, count, spans, limits, expected in (
            ('doc308', 997, [(3, 985)], [996, 3], 0),
            ('text32k', 32766, [], [32765], 32512),
            ('whole_image_then_text', 1500, [(3, 985)], [1499], 1280),
            ('two_images', 1999, [(3, 1027), (1050, 1900)], [1998, 1050, 3], 0)):
        request = Request(request_id=label, prompt_token_ids=[1]*count,
            sampling_params=SamplingParams(max_tokens=1, temperature=0), pooling_params=None,
            mm_features=[MultiModalFeatureSpec(data=None, modality='image', identifier=f'{label}-{i}',
                mm_position=PlaceholderRange(offset=a, length=b-a)) for i, (a, b) in enumerate(spans)])
        calls = []
        def lookup(hashes, limit):
            calls.append(limit)
            return ([],), limit//256*256, 0
        manager = object.__new__(KVCacheManager)
        manager.enable_caching = True
        manager.enable_kv_cache_events = False
        manager.empty_kv_cache_blocks = KVCacheBlocks(([],))
        manager.coordinator = NS(find_longest_cache_hit=lookup)
        _, hit, _ = manager.get_computed_blocks(request)
        assert calls == limits and hit == expected, (label, calls, hit)
        records.append(dict(case=label, limits=calls, hit=hit))
    assert not torch.cuda.is_initialized()
    return dict(pid=os.getpid(), source_file=vision.__file__, source_sha256=OVERRIDE_SHA,
        torch_cuda_initialized=False, native_registration=True,
        entrypoint_bootstrap_executed=True, cases=records)


def child(queue):
    try:
        queue.put(dict(ok=True, result=check()))
    except BaseException as error:
        queue.put(dict(ok=False, error=f'{type(error).__name__}: {error}'))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    if output.parent != Path('/output') or output.exists() or output.is_symlink():
        raise ValueError('Dedicated fresh /output report required')
    limits = {key:(Path('/sys/fs/cgroup')/key).read_text().strip()
              for key in ('memory.max', 'memory.swap.max', 'cpu.max')}
    if limits['memory.max'] != str(2*2**30) or limits['memory.swap.max'] != '0':
        raise RuntimeError('Require actual2GiB/no-swap CPU qualification cgroup')
    parent = check()
    context = multiprocessing.get_context('spawn')
    queue = context.Queue()
    process = context.Process(target=child, args=(queue,))
    process.start()
    try:
        result = queue.get(timeout=120)
        process.join(timeout=20)
        if not result['ok'] or process.exitcode != 0:
            raise RuntimeError(f'Native spawn qualification failed: {result}, exit={process.exitcode}')
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
        if process.is_alive():
            process.kill()
            process.join(timeout=10)
        queue.close()
    report = dict(status='native_image_prefix_parent_spawn_cpu_pass', parent=parent,
        child=result['result'], limits=limits, gpu_devices_exposed=False,
        weights_loaded=False, original_image_modified=False,
        probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with output.open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
