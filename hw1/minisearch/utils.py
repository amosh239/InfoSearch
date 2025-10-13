import re

def wildcard_to_regex(pattern: str) -> re.Pattern:
    esc = ''.join('\\' + c if c in '.^$+{}[]()|\\' else c for c in pattern)
    esc = esc.replace('*', '.*').replace('?', '.')
    return re.compile('^' + esc + '$')

def edit_distance(a: str, b: str, max_d: int = 2) -> int:
    if abs(len(a) - len(b)) > max_d:
        return max_d + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        start = max(1, i - max_d)
        end = min(len(b), i + max_d)
        for j in range(1, start):
            cur.append(max_d + 1)
        for j in range(start, end + 1):
            cost = 0 if ca == b[j - 1] else 1
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + cost))
        for j in range(end + 1, len(b) + 1):
            cur.append(max_d + 1)
        prev = cur
        if min(prev) > max_d:
            return max_d + 1
    return prev[-1]
