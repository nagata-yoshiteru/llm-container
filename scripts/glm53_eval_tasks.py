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
