# SPDX-License-Identifier: AGPL-3.0-only
"""Extra BF16 distribution and real-vocabulary graph checks for native block verification.

Run in the selected runtime image with no model workers loaded. Uses the
installed project's test helpers, without modifying its tests or kernels.
"""
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import torch
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample


def main():
    path = Path('/opt/vllm-v41/tests/v1/spec_decode/test_rejection_sampler_utils.py')
    spec = importlib.util.spec_from_file_location('native_tests', path)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    results = []
    for temp in (.6, 1.):
        torch.manual_seed(182)
        target = torch.randn(4096, device='cuda') / temp
        draft = torch.randn(4096, device='cuda').bfloat16()
        inputs = native._build_rejection_sample_inputs(target, draft, 3, temp, 40960)
        output, counts = rejection_sample(**inputs, num_speculative_steps=3, use_block_verification=True)
        for pos in range(4):
            native._assert_distribution_match(output[counts > pos, pos], target.softmax(0),
                'cuda', label=f'bf16_temp{temp}_pos{pos}')
        results.append(dict(test='bf16_distribution', temperature=temp, trials=40960, passed=True))
        del inputs, output, counts, target, draft
        torch.cuda.empty_cache()
    for requests in (1, 6):
        for temp in (0., 1.):
            torch.manual_seed(93)
            target = torch.randn(129280, device='cuda')
            draft = torch.randn(129280, device='cuda').bfloat16()
            inputs = native._build_rejection_sample_inputs(target, draft, 3, temp, requests)
            # Request-state order need not match the active batch order.
            mapping = torch.arange(requests - 1, -1, -1, device='cuda', dtype=torch.int32)
            inputs['idx_mapping'] = mapping
            inputs['expanded_idx_mapping'] = mapping.repeat_interleave(4)
            inputs['draft_sampled'].view(requests, 4)[-1, -1] = -1
            def call():
                return rejection_sample(**inputs, num_speculative_steps=3, use_block_verification=True)
            for _ in range(3):
                call()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured, captured_counts = call()
            for changed in (False, True):
                if changed:
                    inputs['target_logits'].copy_(inputs['target_logits'].roll(23, dims=-1))
                    inputs['draft_sampled'].view(requests, 4)[:, 1] = 512
                expected, counts = call()
                graph.replay()
                torch.cuda.synchronize()
                assert torch.equal(counts, captured_counts)
                mask = torch.arange(4, device='cuda')[None, :] < counts[:, None]
                assert torch.equal(expected[mask], captured[mask])
                assert torch.all((captured[mask] >= 0) & (captured[mask] < 129280))
                if temp == 0.:
                    assert torch.equal(captured[:, 0], inputs['target_logits'].view(requests, 4, -1)[:, 0].argmax(-1))
                results.append(dict(test='graph_replay', requests=requests, temperature=temp,
                    changed_inputs=changed, passed=True))
            del graph, inputs, captured, captured_counts, expected, counts, target, draft
            torch.cuda.empty_cache()
    source = Path(inspect.getfile(rejection_sample))
    print(json.dumps(dict(status='passed', cases=results, device=torch.cuda.get_device_name(),
        verifier_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        native_tests_sha256=hashlib.sha256(path.read_bytes()).hexdigest())), flush=True)


if __name__ == '__main__':
    main()
