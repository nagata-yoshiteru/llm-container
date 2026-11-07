#!/usr/bin/env python3
"""glm53_eval_tasks.py の難問セット (HARD_MATH / HARD_CODE) の正解とテストを検証する。

    venv-vmlx/bin/python -I scripts/verify-hard-tasks.py   # (標準ライブラリだけで動く)

- 数学: 各問の正解を 2 通りの方法 (総当たりと DP など) で計算し直し、HARD_MATH の値と照合する。
- コード: 各問のテストを、総当たりと照合済みの参照実装に対して実行する
  (テスト自体が正しく、時間内に通ることの確認)。参照実装はモデルには見せない。
"""
import sys
import time
from collections import deque
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from glm53_eval_tasks import HARD_CODE, HARD_MATH  # noqa: E402

sys.setrecursionlimit(10000)
MOD = 10**9 + 7


# ---------------------------------------------------------------- 数学 (2 通りずつ)
def hm1_search(R=4, C=10):
    @lru_cache(None)
    def go(filled):
        for c in range(C):
            for r in range(R):
                i = c * R + r
                if not (filled >> i) & 1:
                    total = 0
                    if r + 1 < R and not (filled >> (i + 1)) & 1:
                        total += go(filled | (1 << i) | (1 << (i + 1)))
                    if c + 1 < C and not (filled >> (i + R)) & 1:
                        total += go(filled | (1 << i) | (1 << (i + R)))
                    if r + 1 < R and c + 1 < C:
                        cells = [i, i + 1, i + R, i + R + 1]
                        if all(not (filled >> k) & 1 for k in cells):
                            total += go(filled | sum(1 << k for k in cells))
                    return total
        return 1
    return go(0)


def hm1_profile(R=4, C=10):
    def fills(r, cur, nxt, out):
        if r == R:
            out.append(nxt); return
        if cur >> r & 1:
            fills(r + 1, cur, nxt, out); return
        if r + 1 < R and not cur >> (r + 1) & 1:
            fills(r + 2, cur | 3 << r, nxt, out)
            fills(r + 2, cur | 3 << r, nxt | 3 << r, out)
        fills(r + 1, cur | 1 << r, nxt | 1 << r, out)
    dp = {0: 1}
    for _ in range(C):
        nd = {}
        for m, w in dp.items():
            out = []; fills(0, m, 0, out)
            for n in out: nd[n] = nd.get(n, 0) + w
        dp = nd
    return dp.get(0, 0)


def hm2_brute():
    return sum(1 for mask in range(1 << 20) if not mask & (mask >> 1)
               and sum(i + 1 for i in range(20) if mask >> i & 1) % 7 == 0)


def hm2_dp():
    dp = {(0, 0): 1}
    for i in range(1, 21):
        nd = {}
        for (last, s), w in dp.items():
            nd[(0, s)] = nd.get((0, s), 0) + w
            if not last:
                k = (1, (s + i) % 7); nd[k] = nd.get(k, 0) + w
        dp = nd
    return sum(w for (_, s), w in dp.items() if s == 0)


def hm3_brute():
    return sum(1 for n in range(1, 10**6 + 1) if (n * n + 1) % 1625 == 0)


def hm3_crt():
    res = [x for x in range(1625) if (x * x + 1) % 1625 == 0]
    q, r = divmod(10**6, 1625)
    return q * len(res) + sum(1 for x in res if 1 <= x <= r)


def hm4_pow():
    return sum(n for n in range(1, 10001) if (pow(2, n, n) + 1) % n == 0)


def hm4_bigint():
    return sum(n for n in range(1, 10001) if (2**n + 1) % n == 0)


def hm5_kirchhoff(R=3, C=5):
    V = R * C
    L = [[Fraction(0)] * V for _ in range(V)]
    for r in range(R):
        for c in range(C):
            u = r * C + c
            for dr, dc in ((0, 1), (1, 0)):
                if r + dr < R and c + dc < C:
                    v = (r + dr) * C + c + dc
                    L[u][u] += 1; L[v][v] += 1; L[u][v] -= 1; L[v][u] -= 1
    M = [row[1:] for row in L[1:]]
    n = len(M); det = Fraction(1)
    for i in range(n):
        p = next(k for k in range(i, n) if M[k][i] != 0)
        if p != i:
            M[i], M[p] = M[p], M[i]; det = -det
        det *= M[i][i]
        for k in range(i + 1, n):
            f = M[k][i] / M[i][i]
            for j in range(i, n): M[k][j] -= f * M[i][j]
    return int(det)


def hm5_enumerate(R=3, C=5):
    # 辺を 14 本選ぶ組合せのうち、閉路を作らないもの (union-find) を数える
    from itertools import combinations
    V = R * C
    edges = [(r * C + c, (r + dr) * C + c + dc) for r in range(R) for c in range(C)
             for dr, dc in ((0, 1), (1, 0)) if r + dr < R and c + dc < C]
    count = 0
    for chosen in combinations(range(len(edges)), V - 1):
        parent = list(range(V))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        ok = True
        for i in chosen:
            a, b = find(edges[i][0]), find(edges[i][1])
            if a == b: ok = False; break
            parent[a] = b
        count += ok
    return count


def hm6_memo():
    @lru_cache(None)
    def E(t):
        return Fraction(0) if t >= 10 else 1 + sum(E(t + k) for k in range(1, 7)) / 6
    e = E(0); return e.numerator + e.denominator


def hm6_bottom_up():
    E = [Fraction(0)] * 16
    for t in range(9, -1, -1):
        E[t] = 1 + sum(E[t + k] for k in range(1, 7)) / 6
    return E[0].numerator + E[0].denominator


MATH_CHECKS = {
    "HM1_tiling": (hm1_search, hm1_profile),
    "HM2_subsets": (hm2_brute, hm2_dp),
    "HM3_crt": (hm3_brute, hm3_crt),
    "HM4_divides": (hm4_pow, hm4_bigint),
    "HM5_spanning": (hm5_kirchhoff, hm5_enumerate),
    "HM6_dice": (hm6_memo, hm6_bottom_up),
}


# ---------------------------------------------------------------- コード (参照実装)

def ref_distinct(s):
    link, length, nxt = [-1], [0], [{}]
    last = 0
    for ch in s:
        cur = len(length); length.append(length[last] + 1); link.append(-1); nxt.append({})
        p = last
        while p != -1 and ch not in nxt[p]:
            nxt[p][ch] = cur; p = link[p]
        if p == -1: link[cur] = 0
        else:
            q = nxt[p][ch]
            if length[p] + 1 == length[q]: link[cur] = q
            else:
                cl = len(length); length.append(length[p] + 1); link.append(link[q]); nxt.append(dict(nxt[q]))
                while p != -1 and nxt[p].get(ch) == q:
                    nxt[p][ch] = cl; p = link[p]
                link[q] = link[cur] = cl
        last = cur
    return sum(length[i] - length[link[i]] for i in range(1, len(length)))

def ref_process(a, ops):
    n = len(a); size = 1
    while size < max(1, n): size *= 2
    s1 = [0] * (2 * size); s2 = [0] * (2 * size); lz = [0] * (2 * size); ln = [0] * (2 * size)
    for i in range(n): s1[size + i] = a[i] % MOD; s2[size + i] = a[i] * a[i] % MOD; ln[size + i] = 1
    for i in range(size - 1, 0, -1):
        s1[i] = (s1[2*i] + s1[2*i+1]) % MOD; s2[i] = (s2[2*i] + s2[2*i+1]) % MOD; ln[i] = ln[2*i] + ln[2*i+1]
    def apply(i, v):
        s2[i] = (s2[i] + 2 * v * s1[i] + v * v % MOD * ln[i]) % MOD
        s1[i] = (s1[i] + v * ln[i]) % MOD
        lz[i] = (lz[i] + v) % MOD
    def push(i):
        if lz[i]: apply(2*i, lz[i]); apply(2*i+1, lz[i]); lz[i] = 0
    def upd(i, lo, hi, l, r, v):
        if r < lo or hi < l: return
        if l <= lo and hi <= r: apply(i, v); return
        push(i); mid = (lo + hi) // 2
        upd(2*i, lo, mid, l, r, v); upd(2*i+1, mid+1, hi, l, r, v)
        s1[i] = (s1[2*i] + s1[2*i+1]) % MOD; s2[i] = (s2[2*i] + s2[2*i+1]) % MOD
    def qry(i, lo, hi, l, r):
        if r < lo or hi < l: return 0
        if l <= lo and hi <= r: return s2[i]
        push(i); mid = (lo + hi) // 2
        return (qry(2*i, lo, mid, l, r) + qry(2*i+1, mid+1, hi, l, r)) % MOD
    out = []
    for op in ops:
        if op[0] == 'add': upd(1, 0, size - 1, op[1], op[2], op[3] % MOD)
        else: out.append(qry(1, 0, size - 1, op[1], op[2]))
    return out

def ref_pairs(n, edges, k):
    adj = [[] for _ in range(n)]
    for u, v in edges: adj[u].append(v); adj[v].append(u)
    removed = [False] * n; sz = [0] * n; total = 0
    def subtree_sizes(root):
        order = []; parent = {root: -1}; st = [root]
        while st:
            u = st.pop(); order.append(u)
            for w in adj[u]:
                if not removed[w] and w != parent[u]: parent[w] = u; st.append(w)
        for u in reversed(order):
            sz[u] = 1 + sum(sz[w] for w in adj[u] if not removed[w] and w != parent[u])
        return order, parent
    def centroid(root):
        order, parent = subtree_sizes(root); tot = sz[root]
        for u in order:
            mx = tot - sz[u]
            for w in adj[u]:
                if not removed[w] and w != parent[u]: mx = max(mx, sz[w])
            if mx * 2 <= tot: return u
    stack = [0]
    while stack:
        c = centroid(stack.pop()); cnt = [1]  # depth counts in already processed branches (depth 0 = centroid)
        for w in adj[c]:
            if removed[w]: continue
            depths = []; q = deque([(w, c, 1)])
            while q:
                u, p, d = q.popleft()
                if d > k: continue
                depths.append(d)
                for x in adj[u]:
                    if x != p and not removed[x] and x != c: q.append((x, u, d + 1))
            for d in depths:
                if 0 <= k - d < len(cnt): total += cnt[k - d]
            for d in depths:
                while len(cnt) <= d: cnt.append(0)
                cnt[d] += 1
        removed[c] = True
        for w in adj[c]:
            if not removed[w]: stack.append(w)
    return total

def ref_count(L, R, K):
    def f(N):  # count x in [1, N] with no two adjacent equal digits and x % K == 0
        if N <= 0: return 0
        ds = list(map(int, str(N))); from functools import lru_cache
        @lru_cache(None)
        def go(pos, prev, rem, tight, started):
            if pos == len(ds): return 1 if started and rem == 0 else 0
            lim = ds[pos] if tight else 9; tot = 0
            for d in range(lim + 1):
                if started and d == prev: continue
                ns = started or d != 0
                tot += go(pos + 1, d if ns else -1, (rem * 10 + d) % K if ns else 0, tight and d == lim, ns)
            return tot
        return go(0, -1, 0, True, False)
    return f(R) - f(L - 1)


REFERENCES = {
    "HC1_substrings": ("distinct_substrings", ref_distinct),
    "HC2_rangesq": ("process", ref_process),
    "HC3_treepairs": ("count_pairs_at_distance", ref_pairs),
    "HC4_digits": ("count_special", ref_count),
}


def main():
    ok = True
    for name, (question, expected) in HARD_MATH.items():
        got = [str(f()) for f in MATH_CHECKS[name]]
        status = "OK" if set(got) == {expected} else "NG"
        ok &= status == "OK"
        print(f"{status} {name}: expected {expected}, recomputed {got}")
    for name, (_, test) in HARD_CODE.items():
        func_name, ref = REFERENCES[name]
        env = {func_name: ref}
        t = time.time()
        try:
            exec(test, env)
            print(f"OK {name}: reference passes the test ({time.time() - t:.1f}s)")
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"NG {name}: {exc!r}")
    print("ALL OK" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

