# compression.py
from typing import List

def encode_vbyte(number: int) -> bytes:
    """Кодирует одно число в VByte."""
    out = []
    while True:
        byte = number & 0x7F
        number >>= 7
        if number == 0:
            out.append(byte | 0x80)
            break
        out.append(byte)
    return bytes(out)

def decode_vbyte(stream: bytes):
    """Генератор, декодирующий поток байт в числа."""
    n = 0
    shift = 0
    for byte in stream:
        n |= (byte & 0x7F) << shift
        if byte & 0x80:
            yield n
            n = 0
            shift = 0
        else:
            shift += 7

def compress_posting_list(ids: List[int]) -> bytes:
    """Применяет Delta-кодирование и затем VByte."""
    if not ids:
        return b""
    # Сортируем (на всякий случай) и считаем дельты
    sorted_ids = sorted(ids)
    deltas = [sorted_ids[0]]
    for i in range(1, len(sorted_ids)):
        deltas.append(sorted_ids[i] - sorted_ids[i-1])
    
    # Сжимаем дельты
    buffer = bytearray()
    for d in deltas:
        buffer.extend(encode_vbyte(d))
    return bytes(buffer)

def decompress_posting_list(data: bytes) -> List[int]:
    """Декодирует VByte и восстанавливает числа из дельт."""
    deltas = list(decode_vbyte(data))
    if not deltas:
        return []
    
    ids = [deltas[0]]
    for i in range(1, len(deltas)):
        ids.append(ids[-1] + deltas[i])
    return ids