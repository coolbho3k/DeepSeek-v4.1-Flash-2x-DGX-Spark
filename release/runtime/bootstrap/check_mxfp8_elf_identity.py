"""CPU-only corruption checks for the debug-stripping runtime identity."""
from pathlib import Path
import struct
import prepare_mxfp8_aot as elf


def main():
    raw=(elf.ROOT/'artifacts/ds41-mxfp8-clean-bootstrap-v2-cache'/elf.REL/(elf.MODULE+'.so')).read_bytes()
    identity=elf.runtime_identity(raw)
    h=struct.unpack_from('<16sHHIQQQIHHHHHH',raw)
    sections=[struct.unpack_from('<IIQQQQIIQQ',raw,h[6]+i*h[11]) for i in range(h[12])]
    count=0
    for section in sections:
        if section[2]&2 and section[1]!=8 and section[5]:
            bad=bytearray(raw);bad[section[4]]^=1
            try:different=elf.runtime_identity(bad)!=identity
            except (ValueError,UnicodeError):different=True
            assert different;count+=1
    bad=bytearray(raw);bad[h[5]]^=1
    assert elf.runtime_identity(bad)!=identity
    for bad in (b'',raw[:1024],b'not ELF'+raw[7:]):
        try:elf.runtime_identity(bad)
        except (ValueError,struct.error):pass
        else:raise AssertionError('Bad/truncated ELF accepted')
    print(f'PASS: {count} allocated-section corruptions and program-header corruption detected;3 invalid ELF refusals; no binary writes/GPU')


if __name__=='__main__':main()
