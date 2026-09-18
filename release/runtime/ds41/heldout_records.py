"""Frozen baseline held-out records and verified next-token target boundaries.

No calibration fitting or candidate selection is performed here. Image answer
loss is teacher-forced; it must never be reported as generated-answer accuracy.
"""
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import PreTrainedTokenizerFast

from .reference_runtime import ReferenceRuntime, REVISION

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def tensor_digest(tensor):
    return hashlib.sha256(tensor.cpu().contiguous().numpy().tobytes()).hexdigest()


def target_masks(tokens, types, answer_start=None, answer_end=None):
    """Masks are indexed by prediction position; target is tokens[position+1]."""
    if tokens.ndim != 1 or tokens.dtype != torch.int64 or types.shape != tokens.shape or types.dtype != torch.int64:
        raise ValueError('Expected same-length I64 token and type vectors')
    if len(tokens) < 2 or (tokens < 0).any() or (types < -1).any():
        raise ValueError('Invalid token sequence or image type values')
    masks = {'all_text_targets': types[1:] == -1}
    if (answer_start is None) != (answer_end is None):
        raise ValueError('Answer boundaries must be provided together')
    if answer_start is not None:
        if not 1 <= answer_start < answer_end == len(tokens) - 1:
            raise ValueError('Expected a nonempty answer followed by one EOS token')
        if (types[answer_start:] != -1).any():
            raise ValueError('Image token appeared in assistant answer targets')
        target_positions = torch.arange(1, len(tokens), device=tokens.device)
        masks['answer_only'] = (target_positions >= answer_start) & (target_positions < answer_end)
        masks['answer_with_eos'] = target_positions >= answer_start
    return masks


class HeldoutRecords:
    def __init__(self, root=ROOT):
        self.root = Path(root).resolve()
        self.capture = self.root / 'calibration/capture-source-v1'
        self.corpus = self.root / 'calibration/corpus-v1'
        self.identity_path = self.capture / 'capture-config.json'
        self.identity = json.loads(self.identity_path.read_text())
        if (self.identity['source_revision'] != REVISION
                or self.identity['method'] != 'source_layerwise_reference_v1'
                or self.identity['max_seq_len'] != 2048):
            raise ValueError('Expected the frozen full baseline capture identity')
        required = {'scripts/capture_reference.py', 'ds41/reference_runtime.py', 'ds41/ssd_rows.py'}
        if set(self.identity['implementation_sha256']) != required:
            raise ValueError('Unexpected capture implementation identity')
        for relative, expected in self.identity['implementation_sha256'].items():
            if digest(self.root / relative) != expected:
                raise ValueError(f'Frozen source capture code changed: {relative}')
        records_path = self.corpus / 'records.jsonl'
        if digest(records_path) != self.identity['corpus_sha256']:
            raise ValueError('Frozen corpus differs from capture')
        all_records = [json.loads(line) for line in records_path.read_text().splitlines()]
        if [r['id'] for r in all_records] != self.identity['records'] or len(all_records) != 384:
            raise ValueError('Expected the entire baseline corpus in captured order')
        if len({r['id'] for r in all_records}) != len(all_records):
            raise ValueError('Duplicate record IDs')
        if any(Path(r['id']).name != r['id'] or r['id'] in ('.', '..') for r in all_records):
            raise ValueError('Unsafe record ID')
        self.records = [r for r in all_records if r['split'] == 'heldout']
        if (sum(r['kind'] == 'text' for r in self.records), sum(r['kind'] == 'image' for r in self.records)) != (40, 24):
            raise ValueError('Expected all40 text and24 image held-out records')
        verified = json.loads((self.root / 'reports/source-verification.json').read_text())
        if (verified['revision'] != REVISION or not verified['complete']
                or len(verified['shards']) != 48 or any(s['status'] != 'verified' for s in verified['shards'])):
            raise ValueError('Pinned source checkpoint has not been fully verified')
        self.runtime = ReferenceRuntime(self.root / 'source' / REVISION, 2048)
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(self.runtime.source / 'tokenizer.json'))

    def verify_head_shards(self):
        verified = json.loads((self.root / 'reports/source-verification.json').read_text())
        expected = {item['file']: item['expected_sha256'] for item in verified['shards']}
        names = {self.runtime.weights.index[key] for key in ('head.weight', 'norm.weight')}
        result = {}
        for name in sorted(names):
            if Path(name).name != name or name not in expected:
                raise ValueError('Output head references an unexpected source shard')
            sha = digest(self.runtime.source / name)
            if sha != expected[name]:
                raise ValueError('Source output-head/norm shard checksum changed')
            result[name] = sha
        return result

    def validate_final_capture(self):
        """Fail closed while ANY transformer layer is still incomplete."""
        for layer in range(40):
            path = self.capture / 'coverage' / f'{layer:02d}.json'
            if not path.exists():
                raise ValueError(f'Source final-state scoring unavailable: layer{layer} is incomplete')
            coverage = json.loads(path.read_text())
            if coverage['layer'] != layer or coverage['records'] != 384:
                raise ValueError(f'Incomplete source layer{layer} coverage')
        expected = {r + '.safetensors' for r in self.identity['records']}
        if {p.name for p in (self.capture / 'states/40').glob('*.safetensors')} != expected:
            raise ValueError('Final source state inventory is incomplete')

    def reconstruct(self, record):
        if record not in self.records:
            raise ValueError('Only the frozen baseline held-out records are eligible')
        answer_start = answer_end = None
        if record['kind'] == 'text':
            tokens = torch.tensor(record['tokens'], dtype=torch.int64)
            types = torch.full_like(tokens, -1)
        else:
            image_path = (self.corpus / record['image']).resolve()
            if not image_path.is_relative_to(self.corpus.resolve()) or digest(image_path) != record['image_sha256']:
                raise ValueError('Image source changed or escapes the corpus')
            user = {'role': 'user', 'content': [
                {'type': 'image_url', 'image_url': {'url': str(image_path)}},
                {'type': 'text', 'text': record['question']}]}
            encode = self.runtime.encoding.encode_messages
            prefix, prefix_media = encode([user], thinking_mode='chat', return_multi_modal_data=True)
            full, media = encode([user, {'role': 'assistant', 'content': record['answer']}],
                                 thinking_mode='chat', return_multi_modal_data=True)
            eos = self.runtime.encoding.eos_token
            if full != prefix + record['answer'] + eos or prefix_media != media:
                raise ValueError('Official assistant prefix/body/EOS composition changed')
            prepare = self.runtime.image_processor.prepare_vl_inputs
            ids, image_types, _ = prepare(full, media['images'], self.tokenizer, self.runtime.args)
            prefix_ids, prefix_types, _ = prepare(prefix, prefix_media['images'], self.tokenizer, self.runtime.args)
            eos_ids = self.tokenizer.encode(eos)
            if ids[:len(prefix_ids)] != prefix_ids or image_types[:len(prefix_types)] != prefix_types:
                raise ValueError('Assistant boundary changes tokenization or image expansion')
            if len(eos_ids) != 1 or ids[-1] != eos_ids[0]:
                raise ValueError('Expected exactly one trailing EOS token')
            tokens, types = torch.tensor(ids, dtype=torch.int64), torch.tensor(image_types, dtype=torch.int64)
            answer_start, answer_end = len(prefix_ids), len(ids) - 1
            if self.tokenizer.decode(ids[answer_start:answer_end], skip_special_tokens=False) != record['answer']:
                raise ValueError('Answer token slice does not exactly reconstruct the recorded answer')
        path = self.capture / 'states/00' / (record['id'] + '.safetensors')
        with safe_open(path, framework='pt', device='cpu') as state:
            if not torch.equal(tokens, state.get_tensor('tokens')) or not torch.equal(types, state.get_tensor('types')):
                raise ValueError(f'Reconstructed tokenization differs from actual capture: {record["id"]}')
        masks = target_masks(tokens, types, answer_start, answer_end)
        entry = dict(id=record['id'], kind=record['kind'], source=record.get('source', record.get('dataset')),
                     tokens=len(tokens), tokens_sha256=tensor_digest(tokens), types_sha256=tensor_digest(types),
                     answer_start=answer_start, answer_end_exclusive=answer_end,
                     target_counts={name: int(mask.sum()) for name, mask in masks.items()},
                     mask_sha256={name: tensor_digest(mask) for name, mask in masks.items()})
        return tokens, types, masks, entry
