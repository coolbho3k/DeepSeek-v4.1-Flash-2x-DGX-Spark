"""Read-only, post-load inventory for the qualified B12X/packed-wo_a backend.

No tensor values are read and no kernels or serving limits are changed. The
controller inspects a bounded receipt tied to a still-live worker PID/start
time; merely importing B12X or mapping an unused CUTLASS library is not proof.
"""
import functools
import hashlib
import json
import os
from pathlib import Path
import re

ROOT = Path('/cache/ds41-backend-attestation')
MAX_BYTES = 256 * 2**10
AOT_NAME = 'mxfp8_gemm_cutlass_sm120'
AOT_FILE = '/usr/local/lib/python3.12/dist-packages/flashinfer/data/aot/' + AOT_NAME + '/' + AOT_NAME + '.so'
PRIVATE_SOURCES = {'ds41/dcp_communication.py': '7aa4a5e6d978f4be72db26e65619e75c7c09e75a218426afbf4a8aab1d56ce20', 'spark_dcp_communication.py': '3ce8a87e7ab253b0e645c0a0f1f346dd6a0be7d15cb749ee37320d5a65f26f25', 'spark_grouped_prefill.py': 'e09cde421ec879a9e6e34aca2cdd6219f437d78d9cf6512e6dccf4a4f57ae839', 'ds41_miaai_fat_moe_v1.so': '35f11df05fc5b870128db1d521513618a9f7a2ca05953b19a6c1d1194230c097', 'spark_fused_moe_async.py': '5000f91b5b8690a4b08a6220e254035914f8675bb9210787e2b005261c304816', 'miaai_engram.py': '79e771e79820c439478ccb51187b329639eb88e2555d31eed4ade9987cd324e7', 'spark_native_engram.py': 'aa36cb389b9cc0933fce562afe4bc0077ce363c0c9e793130e24e3a1bfb29aaa', 'miaai-row-store-v1.so': 'b66c3eac86ed189277cb456c5d3b866ed494eed346e99449504cb2c7aa8b1f71', 'ds41/dcp_sparse_slots.py': 'acbd5dce12e3a988697268c946f7c1a178cc38a3dc738dbb5a94287b7cc43edb', 'spark_sparse_slots.py': '17794cd1c4641bb70ecbd5027c08b018dd03d984231d8b319263710c46eb3961', 'spark_image_prefix.py': '4f3049b68b0fbdec933c281d0303f2702af729818a94b1af84dbaf0807642ab5', 'vision_inputs_override.py': 'dd2b90570f6f42a70ce3b2b997e0027d98a6baa75889b441ae5c60c6443173aa', 'ds41/fused_sparse_attention.py': '22839ccef76d501a9191b193a522429dcfe98ae9977f31f51be04455ea51d9e9', 'ds41/vllm_fp4_main.py': 'e61a6b984fc351cf5147603035913b17fa4c99d66e654861a124118a04320a36', 'spark_dense_decode_graph.py': '89bb94bf73ebb2859399abc61350794b9cffe205c79d5329b3498e927a60acd2', 'spark_b12x_decode_graph.py': '5b47f0733788c53a3d4b6c1539d5b1564d032f150159c30053cb764fce6fc7fb', 'guarded_worker.py': 'c8798deab9b5a47a4e651ea88844cf0d70fac107e0f7f06f81bb4b0985dd8c48', 'spark_b12x_linear.py': '78d8786397898cca0c987c6470fb055eba6030126ee8ea49592f9cf89598bbd5', 'spark_b12x_fp32_reduce.py': 'a40e457b480899bc768177b2f4be19a97d4cb4562f5912b3077d99bc9684987c', 'spark_packed_wo_a.py': '5bf809bed67f28e2b94b345d039e5e808cb8fcb556d49dc1edfecefe1bbf4fe5', 'spark_indexer_k_math.py': '5ee4f01443af118a6bc50393a967860a30980f7314e38207800a3fbf9840ac57', 'ds41/dcp_head_exchange.py': '9852a3d6794fae5b9fc8aa7ff07d98f00989659998c498850df296f0842ca7c5', 'ds41/online_sparse_attention.py': '0e94b6089877e9e4bcfe855465b75eb064237c1e4f6ab9721efc0a63b00acb9d', 'ds41/mhc_decode_prenorm.py': 'd04b8f42c1dc2415d48b411e8033c9da18ec3f6b8ef5fe74e542c816d3d8d25b', 'ds41/packed_wo_a_tuning.py': '9d29b9666661b4f287a74266821100d7e607972acd33bcaaa2339abf697e3fef', 'ds41/resident_storage_audit.py': 'ba3c40145e031571f0034077d0dbb918f4b4a588b8e75d650b45d9ca111762a6', 'ds41/ssd_vocab_rows.py': 'ccf541d0d0cd1c90a1dcc4fde369c8ec59198da73d7171beb99ad987ffc22736', 'ds41/native_vocab_rows.py': '87f444b230865ff3ea5c5ba88186379da21d4c1c26e7350bdd5dbb8f1f547007', 'ds41/native_vocab_stage.py': 'abad81ff57308b6792baa635a45c427c58576d1bd2a9197f8853134cc3182f09', 'ds41/combined_vocab.py': '1263b823fe93a4209a5d74dbc706a5484dfa21ad1020bf93528f7590e9996058', 'ds41/draft_expert_records.py': 'd409e6009a513a54cb354b603e4ff70ffc775847f95e0bc6cd4ce69b63978a12', 'ds41/safetensor_pack.py': 'bec242a3eb07db82176c4a3258fdebc6c4c2da35bc2b1049d110e51038a7693f', 'ds41/native_draft_records.py': '74661d6fdcb1e72984025930af5acdc382c9f16443c2f7120566f20e16eafa17', 'ds41/native_draft_stage.py': 'd8b15b2041ea8b5323355008edfaafa07c42563c65ee5709bf85fc345d8c6013', 'ds41/dense_register_gemv.py': '6bb115ab1b5123746bda26dab79c98f09fb79294c427c3f51bb9c8ab650d6ee9', 'ds41/staged_decode.py': '600312db003b1a4fe93e552e007f50b9937adbffa93cbd69a0c089fa24c91728', 'ds41/staged_route_prepare.py': 'a39142f73382fe162894f128f4198a79406d3899c3662526b7150aa0d517d146', 'ds41/dense_fused_input.py': '36c11b46603a9b17105946e6bc6a433951ad17f36f5695211d5e8ce7cd546cb9', 'ds41/graph_validation.py': '7aa2124c6adf257695416a557c6d3ab2cf751e0c2e09ed7b914c145dc3598deb', 'ds41/dcp_indexer_graph.py': 'f674975b1f89a1e9ad768d562ac5eac2d16477f515e2447a669477b6f01a1aec', 'ds41/dcp_topk_graph.py': 'c154c1130bf9ba24cdae970a017c4376deef751b7d8a3eef40b6270281bee28e', 'ds41/dcp_candidates_graph.py': '319facb681857ea91fccd150cd5d83cc7b61f3a6352a91299ec212c1bb9f0765', 'ds41/vllm_v2_cache.py': '16ecd91362de27abd91232c0a48865962939017336d57a2c2585dcfee6689575', 'ds41/vllm_owned_graphs.py': '100334aa422bc24f2df8aa5f60d50a717c0ed2ccb28a55ab9f08ca8240fceaa0', 'ds41/combined_config.py': 'b690849dc0df9733c6c5daee860dcc5a471cee5202fec08ddb2f6a5f93aa7a65', 'ds41/combined_dspark.py': '0002c02be83bae5836acbc92edfee61949a5b8b8ff1950106f45447f97f8486b', 'spark_combined_miaai.py': '9afa8ba6b1300681080b5f6f45808b290b26a8e30911943107bd0d67be127a05', 'combined_worker.py': '87184d112fb8c1b592d5fa25fc25702bb112ea6c87f15b5e9f85f8e52755275a', 'libds41_vocab_rows.so': '8bf881eb6ccbfe676907a92a29a7b3a7474995ad6e9ac3f95bd2db3ce2b90903', 'libds41_draft_records.so': 'f4bf4aa832b223f7cde10cdd18808a02a2f9020f723dfe39d7a6969dacb0e361', 'ds41_moe_mul1_v1.so': 'ad46da1ca1d44ecba9d80cb3063e37193e82238672b5c85557dbca5857556066', 'combined-native.json': '24f8f4e8cd839785f97f73b15a9ca29b19e0a5bd1e420df5e306ab7ef35fe007', 'spark_parent_heap.py': '30609eb9e365b8400a2d463e8515e99751dc773a8d767221d547295539c4305e', 'spark_context_free_metadata.py': '41345da109cb4d8492b707c1d078644edb1b50a43f1b3083f2092d76f30c85d1', 'ds41/speculative_prefix_retention.py': 'c183e62741c8bf6176514414d2c80cf2514a5ee7a8f5f9cedccba06a8d3e5f65', 'spark_combined_ready.py': '605064f092620aa53ead5c6a86319aa92c02944492e2ba32967c1e7fde2d7220', 'cooperative_moe.so': 'e85f78abfb74157005f88fb2c2aade5b2b7280d4330be57bfbe8e7aa41c006d9', 'cooperative-native.json': '4ada5d12978558d646fd5ca7f10ac8bf8fa3e7c9c33200421124037c031b75ad', 'ds41/cooperative_contract.py': '7992e34e9ea867d1f8816b48e426ea95a7526356272ca790f65bdea01d6195ee', 'ds41/cooperative_moe.py': '22edf8a5aed2194ac0fd2050a9fd27ffea48dd82de021c40f68fd17109ec6bc5', 'ds41/cooperative_routes.py': 'ae91ace2d4310f9409ade1367b341cadd90bc28b152e355831144f9b09dc7be7', 'ds41/draft_exl3_contract.py': 'e9fb42bc2a928afddb0efe127ff14df55b7fb477bb3001dd0b0473c27019b891', 'ds41/draft_exl3_serving.py': 'aaeab74d2901f81fad52de12e76955a0a4ec43154d4c16c18a7e9e02db1c16a6', 'ds41/launch_profile.py': 'e3145599f8204c86ecd59b3f311fd93dff4a78f98ab43cf125b603bbaddcb039', 'ds41/startup_headroom_override.py': '8313e32ba03888e574744d061e611317df01797a12e2086612bb128d331957a8', 'ds41/display_kv.py': '81ae6b42cb400f5986b853e200eab101323ab992296f43659fc062db985d0c4f', 'libds41_display_kv.so': 'bf60fcdf13126ed74363d2a7f0eff208a7318667d75cde323ba11b6b07f8556c', 'ds41/tokenizer_heap.py': '11d8655f445ac0914a2ee080d04d98d8ff8806e370d6a69ab02b97183c3231ea', 'ds41/online_decode_attention.py': '198b1dc6e54b6980c6db96e465778773d75f334b7dade43f1b73ce71e46abca7', 'ds41/length_aware_topk.py': '83990e153b30aad6f98dd299ec5db57afc85f0654c326455aca535613c1ec20f', 'ds41/length_aware_topk_native.py': 'fc98239082a24a0125ce891ba702618ca28f42072fcd3fb43ea1ad11c8d1d377', 'topk-native/topk.so': '2ee6bde332658f6c3fd0eba299a6ed9cbdc3a72b78f17fb9c7bd149d0246f672', 'topk-native/complete.json': '45069f5ffa36ace0ad5bd945d69f311d3a49a1259f524c2d453d64768bad6645', 'ds41/dcp_overlap/__init__.py': '60cd06da8801cf7403d06e37398b92409f2bab7c9dea47744ba56df72b83599d', 'ds41/dcp_overlap/policy.py': '74d7e3f8527f9079110f8033dde63ed9b0c2d3b135cb1035e043b259e4f2d806', 'ds41/dcp_overlap/transport.py': '10f1e338b31db54db30959d127f6bfa8964281aa909f579570297b7015217af3', 'ds41/dcp_overlap/attention.py': '9d186d63cc310c4e3a4fb81448e0bc7b862a2e87d22f69af265ae310a065e466', 'ds41/dcp_overlap/packed.py': '8788de737b1feeda2c4003b27f531547bbf685602d0c28bf6342ab081233b944', 'ds41/dcp_overlap/integration.py': '2b40d6f1c2bcfc204b006c97d5b737c816f34ea1e80954253b5b22c3160e58cb', 'ds41/engram_io/__init__.py': 'fe9dbbb6011b2e725132a30c561aaee713ba155352b16391fe6e5d255117eed0', 'ds41/engram_io/policy.py': '7d67cbc6e419e65d5541e0cc3ce1275a05a1865cbf12d556e99179c4bc3e0254', 'ds41/engram_io/integration.py': '189f4e6dd65ee6095232b7e0e7e6ac33276634c28dd2ebb409d2ff4cd7fbdcb8', 'ds41/engram_io/overlap.py': 'a06b0f946fb625b055fedecd8f0f67cdc3c7fc9c07cf8cd447746d29dd159b17'}
NATIVE_SOURCES = {
    '/opt/ds41-venv/lib/python3.12/site-packages/vllm/model_executor/kernels/linear/mxfp8/b12x.py':
        '7abc42bccf03114e880871fa2ffd67d11466483b2ea636d466e762c17417f3d9',
    '/usr/local/lib/python3.12/dist-packages/b12x/gemm/mxfp8_linear/api.py':
        '04fd4db2a7f4e4ce7f77451e4b03732268104f6707a3b19aa698f92bf2d549d5',
    '/usr/local/lib/python3.12/dist-packages/b12x/gemm/blockscaled/_linear.py':
        'af95f30b04a6ce9c0de8251717721b898163d5d765c4a2b67207672120d38370',
    '/usr/local/lib/python3.12/dist-packages/b12x/_lib/dense_gemm.py':
        'f92d4e1e73a20dd801db200aaa6d89e7463c8b319ac304a19d2d13b194efec4e',
}
_installed = None


def file_sha(path):
    limit = {'ds41_moe_mul1_v1.so': 3266944, 'cooperative_moe.so': 2819968}.get(path.name, 2 * 2**20)
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError('Oversized backend source')
    return hashlib.sha256(raw).hexdigest()


def verify_sources():
    sources = {str(Path(__file__).parent / name): pin for name, pin in PRIVATE_SOURCES.items()}
    sources.update(NATIVE_SOURCES)
    for name, pin in sources.items():
        if file_sha(Path(name)) != pin:
            raise ValueError('Backend source differs from qualification: ' + name)
    return sources


def process_identity(pid, proc=Path('/proc')):
    if type(pid) is not int or pid <= 0:
        raise ValueError('Expected positive worker PID')
    raw = (proc / str(pid) / 'stat').read_text()
    # comm may contain spaces and parentheses; fields after the last ')' start
    # with field 3 (state), making index 19 the field-22 process start time.
    fields = raw[raw.rfind(')') + 2:].split()
    if int(raw.split(' ', 1)[0]) != pid or fields[0] in ('Z', 'X'):
        raise ValueError('Worker is not live')
    return dict(pid=pid, start_time_ticks=int(fields[19]))



def validate_graph_row(row):
    graph = row.get('dense_graph')
    if not isinstance(graph, dict):
        raise ValueError('Loaded B12X layer lacks a graph binding')
    token = graph.get('shared_pool_token')
    if (graph.get('prewarmed') is not True
            or graph.get('input_shape') != [1, row['in_features']]
            or graph.get('output_shape') != [1, row['out_features']]
            or graph.get('input_dtype') != 'torch.bfloat16'
            or graph.get('capture_live_tensor_bytes') != 0
            or graph.get('input_output_bytes') != 2*(row['in_features']+row['out_features'])
            or not isinstance(token, list) or len(token) != 2
            or any(type(v) is not int or v < 0 for v in token) or token == [0, 0]):
        raise ValueError('Invalid loaded B12X graph metadata')
    return tuple(token)


def validate_inventory(inventory):
    if not isinstance(inventory, list) or not 41 <= len(inventory) <= 512:
        raise ValueError('Incomplete or oversized loaded-layer inventory')
    names, grouped, dense = set(), set(), 0
    graph_pools = set()
    for row in inventory:
        name = row['name']
        if not isinstance(name, str) or len(name) > 256 or name in names:
            raise ValueError('Invalid or duplicate loaded-layer name')
        names.add(name)
        if row['backend'] == 'packed_wo_a':
            match = re.search(r'(?:^|\.)layers\.(\d+)\..*\.wo_a$', name)
            if (not match or int(match[1]) in grouped or row['weight_shape'] != [4096, 4096]
                    or row['scale_shape'] != [4096, 128] or row['weight_dtype'] != 'torch.float8_e4m3fn'
                    or row['scale_dtype'] != 'torch.uint8' or row['device'] != 'cuda:0'
                    or row['bmm_batch_size'] != 4 or row['packed_marker'] is not True):
                raise ValueError('Unexpected grouped packed-wo_a storage')
            grouped.add(int(match[1]))
        elif row['backend'] == 'B12X':
            if (row['original_weight_elements'] != 0 or row['original_scale_elements'] != 0
                    or row['device'] != 'cuda:0'
                    or any(type(row[k]) is not int or row[k] <= 0 for k in ('in_features', 'out_features'))):
                raise ValueError('B12X layer has not completed native weight packing')
            graph_pools.add(validate_graph_row(row))
            dense += 1
        else:
            raise ValueError('Unqualified loaded MXFP8 backend')
    if grouped != set(range(40)) or dense == 0 or len(graph_pools) != 1:
        raise ValueError('Expected all forty packed projections and actual B12X linears')
    return dict(packed_wo_a_layers=len(grouped), b12x_layers=dense)


def collect_model(model, base_type, b12x_type, emulation_type):
    if type(model).__name__ != 'DeepseekV41ForCausalLM':
        raise ValueError('Unexpected loaded model')
    inventory = []
    for name, layer in model.named_modules():
        kernel = getattr(getattr(layer, 'quant_method', None), 'kernel', None)
        if not isinstance(kernel, base_type):
            continue
        if getattr(layer, 'is_bmm', False):
            if type(kernel) is not emulation_type:
                raise ValueError('Grouped MXFP8 selection changed: ' + name)
            row = dict(name=name, backend='packed_wo_a', weight_shape=list(layer.weight.shape),
                scale_shape=list(layer.weight_scale.shape), weight_dtype=str(layer.weight.dtype),
                scale_dtype=str(layer.weight_scale.dtype), device=str(layer.weight.device),
                bmm_batch_size=layer.bmm_batch_size, packed_marker=getattr(layer, '_ds41_packed_wo_a', False))
            if str(layer.weight_scale.device) != row['device']:
                raise ValueError('Grouped scale device differs')
        else:
            if type(kernel) is not b12x_type:
                raise ValueError('Non-BMM MXFP8 did not select B12X: ' + name)
            packed = layer.b12x_mxfp8_packed_weight
            row = dict(name=name, backend='B12X', in_features=int(packed.in_features),
                out_features=int(packed.out_features), device=str(packed.weight.values.device),
                original_weight_elements=layer.weight.numel(), original_scale_elements=layer.weight_scale.numel())

            import spark_b12x_decode_graph as graphs
            binding = getattr(layer, graphs.ATTR, None)
            if (type(binding) is not graphs.NativeDenseGraphBinding or graphs._installed is None
                    or binding.graph.pool is not graphs._installed['pool']):
                raise ValueError('Loaded dense layer lacks its qualified prewarmed graph: ' + name)
            binding.validate(layer)
            row['dense_graph'] = binding.describe()
        inventory.append(row)
    validate_inventory(inventory)
    return inventory


ATTENTION_DESCRIPTOR = {'implementation': 'image_safe_packed_sparse_attention_v2', 'kernel_sha256': '22839ccef76d501a9191b193a522429dcfe98ae9977f31f51be04455ea51d9e9', 'registration_sha256': 'e61a6b984fc351cf5147603035913b17fa4c99d66e654861a124118a04320a36', 'query_dtype': 'bfloat16', 'probability_bf16_terms': 2, 'partial_dtype': 'float32', 'lse_base': 2, 'main_cache_format': 'fp4', 'swa_cache_format': 'fp8', 'image_visibility_unchanged': True}


def verify_attention_binding():
    from ds41 import vllm_fp4_main as fp4, fused_sparse_attention as fused
    if (not fp4._installed or fp4._installed_attention_mode != '1'
            or fp4._installed_attention_impl is not fused.packed_sparse_attention_with_lse):
        raise ValueError('Image-safe fused attention was not bound before loading')
    # Validate the original installed native forward and its exact inner
    # callable; this cannot install a missing hook after model loading.
    fp4.register()
    return dict(ATTENTION_DESCRIPTOR)


IMAGE_DESCRIPTOR = {'implementation': 'whole_image_prefix_v1', 'bootstrap_sha256': '4f3049b68b0fbdec933c281d0303f2702af729818a94b1af84dbaf0807642ab5', 'override_sha256': 'dd2b90570f6f42a70ce3b2b997e0027d98a6baa75889b441ae5c60c6443173aa', 'partial_image_prefix_hits': False, 'complete_image_prefix_hits_preserved': True, 'original_image_pixels_preserved': True}


def verify_image_prefix_binding():
    import sys
    import ds41
    from spark_image_prefix import NAME, OVERRIDE_SHA
    vision = sys.modules.get(NAME)
    if (vision is None or getattr(ds41, 'vllm_vision_inputs', None) is not vision
            or getattr(vision, '__file__', None) != str(Path(__file__).parent/'vision_inputs_override.py')
            or getattr(vision, '__ds41_image_prefix_sha256__', None) != OVERRIDE_SHA
            or not getattr(vision, '_registered', False)):
        raise ValueError('Reviewed image-prefix hook was not installed before model loading')
    # Already-installed registration verifies native resize, whole-image
    # scheduling and the actual prefix lookup. Never install a missing hook.
    vision.register()
    return dict(IMAGE_DESCRIPTOR)


SPARSE_MAPPING_DESCRIPTOR = {'implementation': 'stable_fused_dcp2_sparse_slots_v1', 'kernel_sha256': 'acbd5dce12e3a988697268c946f7c1a178cc38a3dc738dbb5a94287b7cc43edb', 'stable_candidate_order': True, 'duplicate_candidates_preserved': True, 'synchronous_bounds_checks': True, 'image_key_membership_unchanged': True, 'maximum_rows': 512, 'maximum_width': 8192, 'persistent_gpu_workspace_bytes': 0}


def verify_sparse_mapping_binding():
    import spark_sparse_slots as slots
    if slots._installed is None:
        raise ValueError('Fused sparse mapper was not installed before model loading')
    slots.register()
    if slots.DESCRIPTOR != SPARSE_MAPPING_DESCRIPTOR:
        raise ValueError('Sparse mapping descriptor changed')
    return dict(SPARSE_MAPPING_DESCRIPTOR)


NATIVE_ENGRAM_DESCRIPTOR = {'implementation': 'miaai_parallel_native_engram_v1', 'license': 'AGPL-3.0-only', 'core_sha256': '79e771e79820c439478ccb51187b329639eb88e2555d31eed4ade9987cd324e7', 'maximum_chunk_tokens': 256, 'maximum_local_heads': 144, 'staging_bytes_per_layer_ceiling': 19759104, 'cache_bytes_per_layer_ceiling': 67108864, 'io_threads': 96, 'resident_tables': False, 'resident_scales': False, 'native_image_hasher_unchanged': True, 'unowned_and_dead_ids_zero': True, 'callback_stream_ordering': True, 'full_model_graph_capture_enabled': True, 'live_tables': 2, 'staging_bounds_verified': True, 'layout': 'page15', 'gpu_readable_host': True, 'deferred_retrieval': True, 'row_bytes_unchanged': True, 'reader_abi': 2}


def verify_native_engram_binding(model):
    import spark_native_engram as hook
    import miaai_engram as core
    if hook._installed is None:
        raise ValueError('Native Engram hook was not installed before loading')
    hook.register()
    stages=[]
    for module in model.modules():
        stage=getattr(module,'_ds41_native_stage',None)
        if stage is None:continue
        if (not isinstance(stage,core.NativeStage) or stage.embedding is not module
                or stage.closed or stage.failed or stage.mode!=('ssd','0','96')):
            raise ValueError('Invalid live native Engram owner')
        cap=module.chunk_tokens*module.part_n_hash_cols
        if (not 0<stage.staging_bytes<=hook.DESCRIPTOR['staging_bytes_per_layer_ceiling']
                or stage.staging_bytes!=cap*(272 if stage.mapped else 536) or stage.ids.numel()!=cap
                or tuple(stage.host_w.shape)!=(cap,256) or tuple(stage.host_s.shape)!=(cap,8)
                or tuple(stage.dev_w.shape)!=(cap,256) or tuple(stage.dev_s.shape)!=(cap,8)
                or not all(t.is_pinned() for t in (stage.ids,stage.host_w,stage.host_s))):
            raise ValueError('Native Engram staging differs from qualification')
        stages.append(stage)
    if len(stages)!=2 or set(stages)!=core._LIVE_STAGES:
        raise ValueError('Exactly two actual native Engram layers are required')
    from ds41.engram_io.integration import audit as audit_engram_io
    audit_engram_io(stages)
    print(json.dumps(dict(stage='ds41_native_engram_loaded',tables=2,
        staging_bytes=sum(s.staging_bytes for s in stages),io_threads=96)),flush=True)
    return dict(NATIVE_ENGRAM_DESCRIPTOR)


GROUPED_PREFILL_DESCRIPTOR = {'implementation': 'miaai_grouped_prefill_ds41_v1', 'license': 'AGPL-3.0-only', 'kernel_sha256': '35f11df05fc5b870128db1d521513618a9f7a2ca05953b19a6c1d1194230c097', 'dispatcher_sha256': 'e09cde421ec879a9e6e34aca2cdd6219f437d78d9cf6512e6dccf4a4f57ae839', 'abi': 1003, 'minimum_expert_rows': 16, 'maximum_tokens': 2048, 'maximum_assignments': 12288, 'shared_workspace_bytes': 280173312, 'workspace_allocated_on_first_large_forward': True, 'small_decode_math_unchanged': True, 'device_only_routing': True, 'pre_down_fp32_routing': True, 'fp16_rounding_boundaries_preserved': True}


def verify_grouped_prefill_binding():
    import spark_grouped_prefill as hook
    import spark_fused_moe as base
    if hook._installed is None:
        raise ValueError('Grouped prefill was not installed before loading')
    hook.register()
    dispatch=base._dispatcher
    if (dispatch is not hook._installed or type(dispatch) is not hook.GroupedDispatcher
            or hook.FAT_MIN!=16 or hook.MAX_ROWS!=12288 or dispatch.failed):
        raise ValueError('Invalid grouped prefill binding')
    if dispatch.fat_workspace is not None and dispatch.fat_workspace.bytes!=280173312:
        raise ValueError('Grouped prefill exceeded its shared scratch contract')
    return dict(GROUPED_PREFILL_DESCRIPTOR)


DCP_COMMUNICATION_DESCRIPTOR = {'implementation': 'fused_dcp2_communication_v1', 'license': 'AGPL-3.0-only', 'kernel_sha256': '7aa4a5e6d978f4be72db26e65619e75c7c09e75a218426afbf4a8aab1d56ce20', 'maximum_rows': 512, 'maximum_sparse_width': 8192, 'stable_sparse_partition': True, 'duplicate_entries_preserved': True, 'packed_output_lse_collective': True, 'per_chunk_collectives': 2, 'sink_exchange_unchanged': True, 'query_exchange_unchanged': True, 'partial_dtype': 'float32', 'lse_base': 2, 'original_image_visibility': True, 'synchronous_cache_bounds_checks': True, 'persistent_gpu_workspace_bytes': 0, 'maximum_packed_send_bytes': 33619968, 'full_model_graph_capture_enabled': False}


def verify_dcp_communication_binding():
    import spark_dcp_communication as hook
    from ds41 import vllm_fp4_main as fp4
    if hook._installed is None or not fp4._installed:
        raise ValueError('Fused DCP communication was not installed before loading')
    hook.register()
    return dict(DCP_COMMUNICATION_DESCRIPTOR)


def record_loaded_worker(worker):
    from vllm.model_executor.kernels import linear
    from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import Mxfp8LinearKernel
    import spark_b12x_linear as selector
    import spark_b12x_fp32_reduce as reducer
    import spark_packed_wo_a as packed
    import spark_indexer_k_math as parity
    import spark_b12x_decode_graph as graphs
    # Their idempotent registrations validate the actual installed bindings;
    # refuse missing hooks instead of silently installing one after loading.
    if any(module._installed is None for module in (selector, reducer, packed, parity, graphs)):
        raise ValueError('Backend hook was not installed before model loading')
    for module in (selector, reducer, packed, parity, graphs):
        module.register()
    if worker.rank not in (0, 1):
        raise ValueError('Expected TP2 worker rank')
    attention = verify_attention_binding()
    image_prefix = verify_image_prefix_binding()
    sparse_mapping = verify_sparse_mapping_binding()
    native_engram = verify_native_engram_binding(worker.model_runner.get_model())
    grouped_prefill = verify_grouped_prefill_binding()
    dcp_communication = verify_dcp_communication_binding()
    inventory = collect_model(worker.model_runner.get_model(), Mxfp8LinearKernel,
        linear.B12xMxfp8LinearKernel, linear.EmulationMxfp8LinearKernel)
    result = dict(format='ds41_loaded_combined_miaai_v9', rank=worker.rank,
        process=process_identity(os.getpid()), sources=verify_sources(),
        observer_sha256=file_sha(Path(__file__)), inventory=inventory,
        counts=validate_inventory(inventory), hooks_verified=True, attention=attention, image_prefix=image_prefix, sparse_mapping=sparse_mapping, native_engram=native_engram, grouped_prefill=grouped_prefill, dcp_communication=dcp_communication)
    raw = (json.dumps(result, sort_keys=True, allow_nan=False) + '\n').encode()
    if len(raw) > MAX_BYTES or ROOT.resolve() != ROOT:
        raise ValueError('Oversized or redirected backend receipt')
    ROOT.mkdir(exist_ok=False)
    with (ROOT / 'worker.json').open('xb') as stream:
        stream.write(raw)
    print(json.dumps(dict(stage='ds41_loaded_backend_verified', rank=worker.rank,
        **result['counts'], receipt_bytes=len(raw))), flush=True)


def register():
    global _installed
    if os.environ.get('DS41_ENABLE_BACKEND_ATTESTATION') != '1':
        raise ValueError('Loaded-backend attestation requires explicit candidate mode')
    import guarded_worker
    verify_sources()
    if _installed is not None:
        if guarded_worker.InitialWorker.load_model is not _installed:
            raise RuntimeError('Loaded-backend observer binding changed')
        return
    original = guarded_worker.InitialWorker.load_model

    @functools.wraps(original)
    def observed(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        record_loaded_worker(self)
        return result

    guarded_worker.InitialWorker.load_model = _installed = observed


def inspect_receipt(rank, expected_sha, root=ROOT, proc=Path('/proc')):
    if rank not in (0, 1) or file_sha(Path(__file__)) != expected_sha:
        raise ValueError('Unqualified backend inspector')
    if root.resolve() != root or not root.is_dir() or sorted(p.name for p in root.iterdir()) != ['worker.json']:
        raise ValueError('Missing, redirected or ambiguous worker receipt')
    path = root / 'worker.json'
    if path.resolve() != path or not path.is_file() or path.stat().st_size > MAX_BYTES:
        raise ValueError('Invalid worker receipt file')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate receipt key')
            result[key] = value
        return result
    result = json.loads(path.read_bytes(), object_pairs_hook=unique)
    if (result['format'] != 'ds41_loaded_combined_miaai_v9' or result['rank'] != rank
            or result['observer_sha256'] != expected_sha or result['hooks_verified'] is not True
            or result.get('attention') != ATTENTION_DESCRIPTOR
            or result.get('image_prefix') != IMAGE_DESCRIPTOR
            or result.get('sparse_mapping') != SPARSE_MAPPING_DESCRIPTOR
            or result.get('native_engram') != NATIVE_ENGRAM_DESCRIPTOR
            or result.get('grouped_prefill') != GROUPED_PREFILL_DESCRIPTOR
            or result.get('dcp_communication') != DCP_COMMUNICATION_DESCRIPTOR
            or result['sources'] != verify_sources()
            or result['counts'] != validate_inventory(result['inventory'])
            or result['process'] != process_identity(result['process']['pid'], proc)):
        raise ValueError('Stale or mismatched loaded-backend evidence')
    mappings = []
    for directory in proc.glob('[0-9]*'):
        try:
            lines = (directory / 'maps').read_text().splitlines()
        except (FileNotFoundError, ProcessLookupError):
            continue
        libraries = sorted({line.split()[-1] for line in lines if AOT_NAME + '.so' in line})
        if libraries:
            if libraries != [AOT_FILE]:
                raise ValueError('Unqualified CUTLASS library mapped alongside B12X')
            mappings.append(dict(pid=int(directory.name), libraries=libraries))
    # Recheck after scanning; never attest a worker that exited mid-inspection.
    if result['process'] != process_identity(result['process']['pid'], proc):
        raise ValueError('Worker changed during backend inspection')
    # Isolated -I/-S inspection: load the hash-verified stdlib-only
    # observer directly, without adding site-packages or the workspace.
    import importlib.util
    ready_path = Path(__file__).with_name('spark_combined_ready.py')
    ready_sha = PRIVATE_SOURCES['spark_combined_ready.py']
    if file_sha(ready_path) != ready_sha:
        raise ValueError('Combined ready observer differs from the frozen kit')
    ready_spec = importlib.util.spec_from_file_location('ds41_ready_inspection', ready_path)
    ready_module = importlib.util.module_from_spec(ready_spec)
    ready_spec.loader.exec_module(ready_module)
    ready = ready_module.inspect_receipt(rank, ready_sha, result['process'],
        process_identity=lambda pid: process_identity(pid, proc))
    return dict(status='qualified_loaded_combined_miaai_with_native_draft_graphs', rank=rank,
        process=result['process'], counts=result['counts'], source_hashes=result['sources'],
        attention=result['attention'], image_prefix=result['image_prefix'], sparse_mapping=result['sparse_mapping'], native_engram=result['native_engram'], grouped_prefill=result['grouped_prefill'], dcp_communication=result['dcp_communication'],
        observer_sha256=expected_sha, optional_cutlass_mappings=mappings, combined_ready=ready)


if __name__ == '__main__':
    import argparse
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (256 * 2**20, 256 * 2**20))
    resource.setrlimit(resource.RLIMIT_CPU, (15, 15))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rank', type=int, choices=(0, 1), required=True)
    parser.add_argument('--source-sha', required=True)
    args = parser.parse_args()
    print(json.dumps(inspect_receipt(args.rank, args.source_sha), sort_keys=True), flush=True)
