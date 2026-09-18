# SPDX-License-Identifier: AGPL-3.0-only
"""Add validated dual-rail wrappers; original node safety checks run first."""
import hashlib
import json
from pathlib import Path


WRAPPERS = '''
# Verified two-rail transport. The original node implementation and every
# RAM, ownership, identity and admission check are retained and called first.
_dual_original_validate_config = validate_config
_dual_original_sample_start = sample_start
_dual_original_validate_start_sample = validate_start_sample
_dual_original_docker_command = docker_command


def validate_config(config):
    _dual_original_validate_config(config)
    if config['fabric_network'] != '10.100.32.0/23':
        raise ValueError('Dual-rail deployment requires the verified two-link subnet')
    for index, node in enumerate(config['nodes']):
        if (node['fabric_ip'] != f'10.100.32.{index+1}'
                or node['ifname'] != 'enp1s0f1np1'
                or node['hca'] != 'rocep1s0f1' or node['gid_index'] != 3):
            raise ValueError('Dual-rail primary topology differs from paired GPU proof')
    return config


def sample_start(node):
    sample = _dual_original_sample_start(node)
    base = Path('/sys/class/infiniband/roceP2p1s0f1/ports/1')
    sample['secondary_rail'] = dict(
        ib_state=(base/'state').read_text().strip(),
        gid=(base/'gids/3').read_text().strip(),
        gid_type=(base/'gid_attrs/types/3').read_text().strip(),
        speed_mbps=int(Path('/sys/class/net/enP2p1s0f1np1/speed').read_text()),
        addresses=json.loads(command(['ip','-j','address','show','dev','enP2p1s0f1np1'])))
    return sample


def validate_start_sample(node, sample):
    _dual_original_validate_start_sample(node, sample)
    rail = sample['secondary_rail']
    address = ipaddress.IPv4Address('10.100.33.'+node['fabric_ip'].rsplit('.',1)[1])
    addresses = {row.get('local') for interface in rail['addresses']
        for row in interface.get('addr_info',[]) if row.get('family') == 'inet'}
    if ('ACTIVE' not in rail['ib_state'] or rail['gid_type'] != 'RoCE v2'
            or rail['speed_mbps'] != 200000 or str(address) not in addresses
            or ipaddress.IPv6Address(rail['gid']).ipv4_mapped != address):
        raise ValueError('Second rail is not the verified active200Gb/s RoCEv2 link')


def docker_command(config, index, owner=None):
    result = _dual_original_docker_command(config, index, owner)
    updates = dict(NCCL_IB_HCA='=rocep1s0f1:1,roceP2p1s0f1:1',
                   NCCL_IB_MERGE_NICS='1')
    for key, value in updates.items():
        before = key+'='+result['env'][key]
        positions = [i for i, item in enumerate(result['command'])
            if i and result['command'][i-1] == '--env' and item == before]
        if len(positions) != 1:
            raise ValueError('Ambiguous dual-rail environment replacement')
        result['command'][positions[0]] = key+'='+value
        result['env'][key] = value
    if (result['env']['NCCL_IB_ADDR_RANGE'] != '10.100.32.0/23'
            or result['env']['NCCL_CROSS_NIC'] != '0'
            or result['env']['NCCL_MAX_CTAS'] != '8'):
        raise ValueError('Dual-rail eight-channel resource profile changed')
    return result


'''


def apply(raw):
    marker = b"if __name__ == '__main__':"
    assert raw.count(marker) == 1
    return raw.replace(marker, WRAPPERS.encode()+marker)


def proof(root):
    root = Path(root)/'reports/native-nccl-protocol-v2'
    pins = {}
    for label in ('single-auto', 'dual-auto'):
        terminal = json.loads((root/f'{label}-terminal.json').read_bytes())
        for host in (0, 1):
            state = terminal[str(host)]
            assert state['ExitCode'] == 0 and not state['OOMKilled'] and not state['Running']
            path = root/f'{label}-host{host}/complete.json'
            raw = path.read_bytes(); result = json.loads(raw)
            assert result['status'] == 'native_two_rank_collectives_pass'
            assert result['rank'] == host and result['protocol'] == '^LL128'
            assert result['dual_rail_requested'] == (label == 'dual-auto')
            assert result['network_environment']['NCCL_MAX_CTAS'] == '8'
            assert result['network_environment']['NCCL_MAX_NCHANNELS'] == '8'
            assert result['limits'] == {'memory.max':'4294967296',
                'memory.swap.max':'0', 'cpu.max':'200000 100000'}
            assert result['allocator_fraction'] == .0075
            assert 0 < result['peak_torch_allocated_bytes'] < 512*2**20
            assert len(result['cases']) == 6 and all(c['exact'] for c in result['cases'])
            traffic = result['rail_data_bytes']['roceP2p1s0f1']
            assert all(v > 10**9 if label == 'dual-auto' else v == 0 for v in traffic.values())
            pins[str(path.relative_to(root))] = hashlib.sha256(raw).hexdigest()
    return pins
