"""Same four math and three coding tasks as the original local GLM benchmark."""

MATH = {
    "M1_aime06": ("Find the number of ordered pairs of positive integers (a, b) such that a + b = 1000 "
                  "and neither a nor b has a zero digit.", "738"),
    "M2_aime99": ("Find the sum of all positive integers n for which n^2 - 19n + 99 is a perfect square.", "38"),
    "M3_digitsum": ("How many integers n with 1 <= n <= 1,000,000 have digit sum exactly 30?", "50877"),
    "M4_powmod": ("What is the remainder when 2^2026 is divided by 1000?", "864"),
}

MATH_SUFFIX = "\n\nGive the final answer as a single integer in the form \\boxed{N}."

CODE = {
    "C1_inversions": (
        "Write a Python function `count_inversions(arr: list[int]) -> int` that returns the number of pairs "
        "(i, j) with i < j and arr[i] > arr[j]. It must run in O(n log n) and handle n = 200000.",
        """
import random, time
def brute(a): return sum(1 for i in range(len(a)) for j in range(i+1,len(a)) if a[i]>a[j])
for _ in range(200):
    a=[random.randint(-5,5) for _ in range(random.randint(0,40))]
    assert count_inversions(list(a))==brute(a),a
a=[random.randint(0,10**9) for _ in range(200000)]
t=time.time(); count_inversions(a); assert time.time()-t<10,'too slow'
"""),
    "C2_domino": (
        "Write a Python function `tilings(n: int) -> int` returning the number of ways to tile a 3 x n board "
        "with 2 x 1 dominoes, modulo 1_000_000_007. n can be as large as 10**18, so it must run in O(log n).",
        """
M=10**9+7
f={0:1,1:0,2:3,3:0}
for k in range(4,60): f[k]=(4*f[k-2]-f[k-4])%M
for k in range(60): assert tilings(k)==f[k],(k,tilings(k),f[k])
def mul(A,B): return [[sum(A[i][t]*B[t][j] for t in range(2))%M for j in range(2)] for i in range(2)]
def ref(n):
    if n%2: return 0
    m=n//2; R=[[1,0],[0,1]]; A=[[4,M-1],[1,0]]
    while m:
        if m&1: R=mul(R,A)
        A=mul(A,A); m>>=1
    return (R[1][0]*3+R[1][1]*1)%M
for k in range(0,60,2): assert ref(k)==f[k]
for n in [10**18, 10**18-1, 123456789012, 999999999999999998]:
    assert tilings(n)==ref(n),n
"""),
    "C3_minwindow": (
        "Write a Python function `min_window(s: str, t: str) -> str` that returns the shortest substring of s "
        "containing every character of t (with multiplicity). If there are several, return the leftmost; "
        "if none, return the empty string. Must be O(len(s) + len(t)).",
        """
import random
def brute(s,t):
    from collections import Counter
    need=Counter(t); best=None
    if not t: return ''
    for i in range(len(s)):
        for j in range(i+1,len(s)+1):
            if not (need-Counter(s[i:j])):
                if best is None or j-i<len(best): best=s[i:j]
                break
    return best or ''
assert min_window('ADOBECODEBANC','ABC')=='BANC'
assert min_window('a','aa')==''
for _ in range(300):
    s=''.join(random.choice('abc') for _ in range(random.randint(0,25)))
    t=''.join(random.choice('abc') for _ in range(random.randint(1,4)))
    assert min_window(s,t)==brute(s,t),(s,t,min_window(s,t),brute(s,t))
"""),
}

CODE_SUFFIX = "\n\nReturn only the final code in a single ```python code block (no tests, no input reading)."


# =============================================================================
# 難問セット (2026-11-07 追加)。--suite hard で使う。
#
# 過去問の暗記で解けないよう新しく作った問題で、正解はすべてプログラムで計算した。
# scripts/verify-hard-tasks.py が、総当たりと別解で正解を計算し直して照合する。
# コード問題の大きいケースの期待値は、総当たりと照合済みの参照実装で計算したもの
# (固定シードの random.Random で同じ入力を作る)。
# =============================================================================

HARD_MATH = {
    "HM1_tiling": ("In how many ways can a 4 x 10 rectangle be tiled completely with 1 x 2 dominoes "
                   "(each placed horizontally or vertically) and 2 x 2 squares? Tiles may not overlap "
                   "or extend outside the rectangle.", "864901"),
    "HM2_subsets": ("How many subsets S of {1, 2, ..., 20} (including the empty set) contain no two "
                    "consecutive integers and have the sum of their elements divisible by 7?", "2531"),
    "HM3_crt": ("How many positive integers n <= 1,000,000 have the property that n^2 + 1 is divisible "
                "by 1625?", "2462"),
    "HM4_divides": ("Find the sum of all positive integers n <= 10,000 such that n divides 2^n + 1.", "29677"),
    "HM5_spanning": ("Consider the 3 x 5 grid graph: its 15 vertices are the points (i, j) with 0 <= i <= 2 "
                     "and 0 <= j <= 4, and two vertices are joined by an edge when they are horizontally or "
                     "vertically adjacent (distance 1). How many spanning trees does this graph have?", "30305"),
    "HM6_dice": ("A fair six-sided die is rolled repeatedly until the running total of all rolls is at "
                 "least 10. The expected number of rolls is m/n, where m and n are relatively prime "
                 "positive integers. Find m + n.", "43572871"),
}

HARD_CODE = {
    "HC1_substrings": (
        "Write a Python function `distinct_substrings(s: str) -> int` that returns the number of distinct "
        "non-empty substrings of s. s consists of lowercase English letters and len(s) <= 200000, so it "
        "must run in about O(n) or O(n log n) (e.g. a suffix automaton, or a suffix array with LCP).",
        """
import random, time
def brute(s): return len({s[i:j] for i in range(len(s)) for j in range(i+1, len(s)+1)})
rng = random.Random(1)
for _ in range(500):
    s = ''.join(rng.choice('abc'[:rng.randint(1, 3)]) for _ in range(rng.randint(0, 30)))
    assert distinct_substrings(s) == brute(s), s
assert distinct_substrings('a' * 200000) == 200000
r = random.Random(20261107); big1 = ''.join(r.choice('ab') for _ in range(200000))
r = random.Random(42); big2 = ''.join(r.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(200000))
t = time.time()
assert distinct_substrings(big1) == 19996800087
assert distinct_substrings(big2) == 19999478344
assert time.time() - t < 20, 'too slow'
"""),
    "HC2_rangesq": (
        "Write a Python function `process(a: list[int], ops: list[tuple]) -> list[int]`. Each element of ops "
        "is either ('add', l, r, v): add v to every a[i] with l <= i <= r, or ('sq', l, r): append the sum of "
        "a[i]**2 over l <= i <= r, taken modulo 1_000_000_007, to the result list. Indices are 0-based and "
        "inclusive, values and v can be negative (|a[i]|, |v| <= 10**9 initially and per update), and the "
        "operations are applied in order. Return the list of answers. len(a) <= 100000 and "
        "len(ops) <= 100000, so it must run in O((n + q) log n).",
        """
import random, time
MOD = 10**9 + 7
def brute(a, ops):
    a = list(a); out = []
    for op in ops:
        if op[0] == 'add':
            for i in range(op[1], op[2] + 1): a[i] += op[3]
        else: out.append(sum(x * x for x in a[op[1]:op[2] + 1]) % MOD)
    return out
def gen(r, n, q, vmax):
    a = [r.randint(-vmax, vmax) for _ in range(n)]; ops = []
    for _ in range(q):
        l = r.randrange(n); rr = r.randrange(l, n)
        ops.append(('add', l, rr, r.randint(-vmax, vmax)) if r.random() < 0.5 else ('sq', l, rr))
    return a, ops
rng = random.Random(2)
for _ in range(300):
    a, ops = gen(rng, rng.randint(1, 12), rng.randint(1, 25), 10**9)
    assert process(list(a), list(ops)) == brute(a, ops), (a, ops)
a, ops = gen(random.Random(7), 100000, 100000, 10**9)
t = time.time(); out = process(a, ops)
assert time.time() - t < 30, 'too slow'
assert len(out) == 49979 and out[:3] == [63705921, 735569381, 147259467]
assert sum(out) % MOD == 235710816 and sum((i + 1) * v for i, v in enumerate(out)) % MOD == 525104876
"""),
    "HC3_treepairs": (
        "Write a Python function `count_pairs_at_distance(n: int, edges: list[tuple[int, int]], k: int) -> int`. "
        "The input is a tree with vertices 0..n-1 and the n-1 given edges. Return the number of unordered pairs "
        "{u, v} of distinct vertices whose distance (number of edges on the path between them) is exactly k. "
        "1 <= n <= 20000 and 1 <= k <= 20000 (k may exceed the diameter). It must be much faster than O(n^2) "
        "(e.g. centroid decomposition), and the tree can be a path of 20000 vertices, so avoid deep recursion.",
        """
import random, time
from collections import deque
def brute(n, edges, k):
    adj = [[] for _ in range(n)]
    for u, v in edges: adj[u].append(v); adj[v].append(u)
    tot = 0
    for s in range(n):
        dist = [-1] * n; dist[s] = 0; q = deque([s])
        while q:
            u = q.popleft()
            for w in adj[u]:
                if dist[w] < 0: dist[w] = dist[u] + 1; q.append(w)
        tot += sum(1 for v in range(s + 1, n) if dist[v] == k)
    return tot
def gen(r, n): return [(i, r.randrange(i)) for i in range(1, n)]
rng = random.Random(3)
for _ in range(300):
    n = rng.randint(1, 40); e = gen(rng, n); k = rng.randint(1, max(1, n))
    assert count_pairs_at_distance(n, list(e), k) == brute(n, e, k), (n, e, k)
t = time.time()
assert count_pairs_at_distance(20000, [(i, i + 1) for i in range(19999)], 7) == 19993
assert count_pairs_at_distance(20000, [(0, i) for i in range(1, 20000)], 2) == 19999 * 19998 // 2
assert count_pairs_at_distance(20000, gen(random.Random(11), 20000), 7) == 1276267
assert count_pairs_at_distance(20000, gen(random.Random(12), 20000), 25) == 2529073
deep = [(i, random.Random(13 + i).randrange(max(0, i - 3), i)) for i in range(1, 20000)]
assert count_pairs_at_distance(20000, deep, 500) == 37943
assert time.time() - t < 60, 'too slow'
"""),
    "HC4_digits": (
        "Write a Python function `count_special(L: int, R: int, K: int) -> int` that returns how many integers x "
        "with L <= x <= R satisfy both conditions: (1) no two adjacent digits in the decimal representation of x "
        "are equal, and (2) x is divisible by K. 1 <= L <= R <= 10**18 and 1 <= K <= 100.",
        """
import random, time
def brute(L, R, K):
    return sum(1 for x in range(L, R + 1) if x % K == 0 and all(a != b for a, b in zip(str(x), str(x)[1:])))
rng = random.Random(4)
for _ in range(300):
    L = rng.randint(1, 5000); R = rng.randint(L, L + rng.randint(0, 5000)); K = rng.randint(1, 100)
    assert count_special(L, R, K) == brute(L, R, K), (L, R, K)
t = time.time()
assert count_special(1, 10**18, 1) == 168856464709124010
assert count_special(1, 10**18, 7) == 24122352101302518
assert count_special(123456789012345, 987654321098765432, 97) == 1737803405706957
assert count_special(10**17, 10**18, 64) == 2284883193084938
assert count_special(999999999999999999, 10**18, 1) == 0
assert time.time() - t < 20, 'too slow'
"""),
}
