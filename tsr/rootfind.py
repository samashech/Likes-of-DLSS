"""Root-finding methods (Experiments 2a-2d), written from scratch.

Each solver returns (root, history), where history is the list of iterates
x_k. Every call to f in our tuning problems is a full upscaler run, so the
number of f evaluations, not iterations, is the real cost. Wrap f in
Counted to measure it.
"""


class Counted:
    """Memoising wrapper that counts distinct evaluations of f."""

    def __init__(self, f):
        self.f = f
        self.cache = {}

    def __call__(self, x):
        key = round(float(x), 12)
        if key not in self.cache:
            self.cache[key] = self.f(x)
        return self.cache[key]

    @property
    def evals(self):
        return len(self.cache)


def bisection(f, a, b, tol=1e-6, max_iter=100):
    """Halve [a, b] keeping the sign change. Always converges; error halves each step."""
    fa = f(a)
    if fa * f(b) > 0:
        raise ValueError("f(a) and f(b) must have opposite signs")
    hist = []
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        hist.append(m)
        fm = f(m)
        if fm == 0 or 0.5 * (b - a) < tol:
            break
        if fa * fm < 0:
            b = m
        else:
            a, fa = m, fm
    return m, hist


def regula_falsi(f, a, b, tol=1e-6, max_iter=100):
    """Like bisection, but split where the chord between (a, fa) and (b, fb)
    crosses zero. Keeps the bracket; can stall with one end stuck."""
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs")
    hist = []
    x_old = None
    for _ in range(max_iter):
        x = b - fb * (b - a) / (fb - fa)
        hist.append(x)
        fx = f(x)
        if fx == 0 or (x_old is not None and abs(x - x_old) < tol):
            break
        if fa * fx < 0:
            b, fb = x, fx
        else:
            a, fa = x, fx
        x_old = x
    return x, hist


def secant(f, x0, x1, tol=1e-6, max_iter=100):
    """Newton's method with the derivative replaced by the slope through the
    last two points. Order ~1.618, no derivative needed, no bracket kept."""
    f0, f1 = f(x0), f(x1)
    hist = [x1]
    for _ in range(max_iter):
        if f1 == f0:
            break
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        hist.append(x2)
        if abs(x2 - x1) < tol:
            break
        x0, f0, x1, f1 = x1, f1, x2, f(x2)
    return hist[-1], hist


def newton(f, df, x0, tol=1e-6, max_iter=100):
    """x <- x - f(x) / f'(x). Quadratic convergence near the root."""
    x = x0
    hist = [x]
    for _ in range(max_iter):
        x_new = x - f(x) / df(x)
        hist.append(x_new)
        if abs(x_new - x) < tol:
            break
        x = x_new
    return hist[-1], hist
