"""Independent exact rational Gauss-Jordan reference; standard library only."""
from fractions import Fraction as F
import random


def solve(matrix, rhs):
    n = len(rhs)
    rows = [[F(x) for x in row] + [F(y)] for row, y in zip(matrix, rhs)]
    for j in range(n):
        pivot = next(i for i in range(j, n) if rows[i][j])
        rows[j], rows[pivot] = rows[pivot], rows[j]
        d = rows[j][j]
        rows[j] = [x / d for x in rows[j]]
        for i in range(n):
            if i != j:
                d = rows[i][j]
                rows[i] = [x - d*y for x, y in zip(rows[i], rows[j])]
    return [row[-1] for row in rows]


def score(matrix, c):
    return -sum(F(x)*y for x, y in zip(c, solve(matrix, c)))


def cases(count=24, seed=18031):
    rng = random.Random(seed)
    for case in range(count):
        p = 1 + case % 4
        q = [[F(rng.randint(-4, 4), 3) for _ in range(p)] for _ in range(p)]
        matrix = [[sum(q[k][i]*q[k][j] for k in range(p)) + (i == j)
                   for j in range(p)] for i in range(p)]
        c = [F(rng.randint(-4, 4), 3) for _ in range(p)]
        c[0] = F(1)
        f = [F(rng.randint(-4, 4), 3) for _ in range(p)]
        yield matrix, c, f
