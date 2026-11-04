"""Correct the cuSPARSELt0.8.1 wheel's SBSA metadata spelling on verified ARM64.

No library bytes are modified. The upstream filename says aarch64 but its WHEEL
metadata says sbsa; uv repeatedly reinstalls it unless that tag is corrected.
"""
import base64
import csv
import hashlib
import importlib.metadata
import platform
import struct


def main():
    if platform.machine()!='aarch64':raise RuntimeError('ARM64 only')
    dist=importlib.metadata.distribution('nvidia-cusparselt-cu13')
    if dist.version!='0.8.1':raise RuntimeError('Review metadata for this version first')
    wheel=next(f for f in dist.files if str(f).endswith('.dist-info/WHEEL'))
    path=dist.locate_file(wheel)
    text=path.read_text()
    old='Tag: py3-none-manylinux2014_sbsa'
    new='Tag: py3-none-manylinux2014_aarch64'
    if old not in text:
        assert new in text
        print('cuSPARSELt ARM64 metadata already correct');return
    libraries=[dist.locate_file(f) for f in dist.files if 'libcusparseLt.so' in str(f)]
    assert libraries
    for library in libraries:
        with library.open('rb') as stream:header=stream.read(20)
        assert header[:4]==b'\x7fELF' and header[5]==1
        assert struct.unpack('<H',header[18:20])[0]==183,'Not an AArch64 library'
    encoded=text.replace(old,new).encode()
    record=next(f for f in dist.files if str(f).endswith('.dist-info/RECORD'))
    record_path=dist.locate_file(record)
    with record_path.open(newline='') as stream:rows=list(csv.reader(stream))
    digest=base64.urlsafe_b64encode(hashlib.sha256(encoded).digest()).rstrip(b'=').decode()
    found=False
    for row in rows:
        if row[0]==str(wheel):row[1:]=['sha256='+digest,str(len(encoded))];found=True
    assert found
    path.write_bytes(encoded)
    with record_path.open('w',newline='') as stream:csv.writer(stream).writerows(rows)
    print('Corrected wheel metadata sbsa -> aarch64; ELF libraries unchanged')


if __name__=='__main__':main()
