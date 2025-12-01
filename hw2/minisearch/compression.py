from typing import List

def compress_posting_list(ids: List[int]) -> bytes:
    out = bytearray()
    last = 0
    for n in sorted(ids):
        delta = n - last
        last = n
        while delta >= 128:
            out.append(delta & 0x7F)
            delta >>= 7
        out.append(delta | 0x80)
    return bytes(out)

def decompress_posting_list(data: bytes) -> List[int]:
    ids = []
    last = n = shift = 0
    for b in data:
        n |= (b & 0x7F) << shift
        if b & 0x80:
            last += n
            ids.append(last)
            n = shift = 0
        else:
            shift += 7
    return ids