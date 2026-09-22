# SPDX-License-Identifier: AGPL-3.0-only
"""Public configuration. Standard library only; importing never touches a GPU."""
import ipaddress
import math
import os
from pathlib import Path
import re

DEFAULTS = {
    'WORKER_HOST': '', 'HEAD_FABRIC_IP': '', 'WORKER_FABRIC_IP': '',
    'FABRIC_NETWORK': '', 'HEAD_IFNAME': '', 'WORKER_IFNAME': '',
    'HEAD_HCA': '', 'WORKER_HCA': '', 'ROCE_GID_INDEX': '3',
    'HEAD_SECONDARY_IP': '', 'WORKER_SECONDARY_IP': '',
    'HEAD_SECONDARY_IFNAME': '', 'WORKER_SECONDARY_IFNAME': '',
    'HEAD_SECONDARY_HCA': '', 'WORKER_SECONDARY_HCA': '',
    'API_HOST': '0.0.0.0', 'API_PORT': '8888', 'MASTER_PORT': '29541',
    'SERVED_MODEL_NAME': 'deepseek-v41-flash-exl3',
    'GPU_MEMORY_UTILIZATION': '0.92', 'MAX_MODEL_LEN': '1048576',
    'MAX_NUM_SEQS': '6', 'MAX_NUM_BATCHED_TOKENS': '2048',
    'LONG_PREFILL_TOKEN_THRESHOLD': '2048',
    'PREFIX_CACHE_RETENTION_INTERVAL': '4096',
    'DS41_FP4_KV_MODE': 'nvfp4_4over6',
    'DS41_SWA_KV_GROUP_SIZE': '32',
    'ALLOW_STARTUP_MEMORY_SHORTFALL': '1',
    'HEAD_DRM_CARD': '/dev/dri/card0', 'WORKER_DRM_CARD': '/dev/dri/card0',
    'DS41_CACHE_DIR': '', 'REMOTE_CACHE_DIR': '', 'REMOTE_DIR': '',
    'EXISTING_DEPLOYMENT': '', 'EXISTING_DEPLOYMENT_SHA256': '',
}


def load(root, overrides=None, environ=None):
    env = os.environ if environ is None else environ
    values = {k: env.get(k, v) for k, v in DEFAULTS.items()}
    for k, v in (overrides or {}).items():
        if v is not None:
            values[k] = str(v)
    missing = [k for k in ('WORKER_HOST', 'HEAD_FABRIC_IP', 'WORKER_FABRIC_IP',
               'FABRIC_NETWORK', 'HEAD_IFNAME', 'WORKER_IFNAME', 'HEAD_HCA', 'WORKER_HCA') if not values[k]]
    if missing:
        raise ValueError('Copy .env.ds41.example to .env.ds41 and set: ' + ', '.join(missing))
    if not re.fullmatch(r'(?:[A-Za-z0-9_][A-Za-z0-9_.-]*@)?[A-Za-z0-9][A-Za-z0-9.-]*', values['WORKER_HOST']):
        raise ValueError('WORKER_HOST must be an SSH hostname/IP, optionally user@host (no shell options)')
    network = ipaddress.IPv4Network(values['FABRIC_NETWORK'], strict=True)
    gid = int(values['ROCE_GID_INDEX'])
    if not 0 <= gid <= 255:
        raise ValueError('ROCE_GID_INDEX must be 0..255')
    rails = []
    for prefix in ('HEAD', 'WORKER'):
        primary = dict(fabric_ip=values[prefix+'_FABRIC_IP'], ifname=values[prefix+'_IFNAME'],
                       hca=values[prefix+'_HCA'], gid_index=gid)
        secondary = [values[prefix+'_SECONDARY_'+k] for k in ('IP', 'IFNAME', 'HCA')]
        if any(secondary) and not all(secondary):
            raise ValueError(prefix + ': set all three SECONDARY settings, or leave all empty')
        node_rails = [primary]
        if all(secondary):
            node_rails.append(dict(fabric_ip=secondary[0], ifname=secondary[1], hca=secondary[2], gid_index=gid))
        for rail in node_rails:
            address = ipaddress.IPv4Address(rail['fabric_ip'])
            if address not in network or address in (network.network_address, network.broadcast_address):
                raise ValueError('Every fabric IP must be a host address inside FABRIC_NETWORK')
            for k in ('ifname', 'hca'):
                if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,31}', rail[k]):
                    raise ValueError('Invalid interface/HCA name: ' + rail[k])
        if len({r['hca'] for r in node_rails}) != len(node_rails):
            raise ValueError('Each rail must use a distinct HCA')
        rails.append(node_rails)
        if not re.fullmatch(r'/dev/dri/card[0-9]+', values[prefix+'_DRM_CARD']):
            raise ValueError('DRM card must be /dev/dri/cardN')
    if len(rails[0]) != len(rails[1]) or len({r['fabric_ip'] for n in rails for r in n}) != sum(map(len, rails)):
        raise ValueError('Use the same number of rails on both hosts and distinct fabric IPs')
    profile = {k: (float(values[k.upper()]) if k == 'gpu_memory_utilization' else int(values[k.upper()]))
               for k in ('gpu_memory_utilization', 'max_model_len', 'max_num_seqs',
                         'max_num_batched_tokens', 'long_prefill_token_threshold',
                         'prefix_cache_retention_interval')}
    if values['DS41_SWA_KV_GROUP_SIZE'] not in ('32', '64'):
        raise ValueError('DS41_SWA_KV_GROUP_SIZE must be 32 or 64')
    profile['swa_kv_group_size'] = int(values['DS41_SWA_KV_GROUP_SIZE'])
    profile['fp4_kv_mode'] = values['DS41_FP4_KV_MODE']
    if profile['fp4_kv_mode'] not in ('nvfp4_4over6', 'legacy'):
        raise ValueError('DS41_FP4_KV_MODE must be nvfp4_4over6 or legacy')
    profile['kv_cap_mib'] = 0  # This release uses the proven display-only KV allocator.
    retention = profile['prefix_cache_retention_interval']
    if not 0 <= retention <= 1048576 or retention % 256:
        raise ValueError('PREFIX_CACHE_RETENTION_INTERVAL must be 0 or a multiple of 256 through 1048576')
    # Do not import worker code to validate user settings.
    util = profile['gpu_memory_utilization']
    if not math.isfinite(util) or not .85 <= util <= .925:
        raise ValueError('GPU_MEMORY_UTILIZATION must be 0.85..0.925; 0.92 is the tested value')
    if not 4096 <= profile['max_model_len'] <= 1048576 or not 1 <= profile['max_num_seqs'] <= 6:
        raise ValueError('MAX_MODEL_LEN must be 4096..1048576; MAX_NUM_SEQS must be 1..6')
    batch = profile['max_num_batched_tokens']
    if batch not in (2048, 3072) or profile['long_prefill_token_threshold'] not in (0, *range(1056, batch+1)):
        raise ValueError('Batch must be 2048 or 3072; long-prefill threshold must be 0 or 1056..batch')
    override = values['ALLOW_STARTUP_MEMORY_SHORTFALL']
    if override not in ('0', '1') or (override == '1' and util != .92):
        raise ValueError('The startup override is scoped to 0.92; set ALLOW_STARTUP_MEMORY_SHORTFALL=0 for other values')
    host = str(ipaddress.IPv4Address(values['API_HOST']))
    port, master = int(values['API_PORT']), int(values['MASTER_PORT'])
    if not all(1024 <= p <= 65535 for p in (port, master)) or port == master:
        raise ValueError('API_PORT and MASTER_PORT must be distinct ports in 1024..65535')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', values['SERVED_MODEL_NAME']):
        raise ValueError('Invalid SERVED_MODEL_NAME')
    cache = Path(values['DS41_CACHE_DIR'] or Path(root)/'.assets').expanduser().resolve()
    for k in ('REMOTE_CACHE_DIR', 'REMOTE_DIR'):
        if values[k] and (not Path(values[k]).is_absolute() or '..' in Path(values[k]).parts):
            raise ValueError(k + ' must be an absolute path, without ~ or ..')
    for path in (str(cache), values['REMOTE_CACHE_DIR'], values['REMOTE_DIR']):
        if path and (len(Path(path).parts) < 3 or any(c in path for c in ('\n', '\r', '\0', ','))):
            raise ValueError('Use a dedicated cache directory, not a filesystem root')
    existing, digest = values['EXISTING_DEPLOYMENT'], values['EXISTING_DEPLOYMENT_SHA256']
    if bool(existing) != bool(digest):
        raise ValueError('Set both EXISTING_DEPLOYMENT and EXISTING_DEPLOYMENT_SHA256, or neither')
    if existing:
        path = Path(existing)
        if (not path.is_absolute() or str(path) != existing or '..' in path.parts
                or any(c in existing for c in ('\n', '\r', '\0'))
                or not re.fullmatch('[0-9a-f]{64}', digest)):
            raise ValueError('Existing deployment requires an absolute path and SHA256 pin')
    return dict(values=values, worker=values['WORKER_HOST'], rails=rails, cache=str(cache),
                api=dict(host=host, port=port, master_port=master, model_name=values['SERVED_MODEL_NAME']),
                serving=profile, fabric_network=str(network), startup_memory_override=override == '1')
