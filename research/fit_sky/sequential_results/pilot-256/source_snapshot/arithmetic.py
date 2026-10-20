"""Arb enclosures and fourth-order directional jets for validated quadrature."""

from math import nextafter, inf

from flint import arb


def ball(lo, hi=None):
    a = arb(float(lo)) if isinstance(lo, float) else arb(lo)
    return a if hi is None else a.union(arb(float(hi)) if isinstance(hi, float) else arb(hi))


def lower(x):
    if x == 0:
        return 0.
    return nextafter(float(x.lower()), -inf)


def upper(x):
    if x == 0:
        return 0.
    return nextafter(float(x.upper()), inf)


def bounds(x):
    return [lower(x), upper(x)]


def square(x):
    if isinstance(x, Jet):
        result = x*x
        result.c[0] = square(x.c[0])
        return result
    x = arb(x)
    return (x*x).nonnegative_part()


class Jet:
    """Taylor coefficients f^(k)/k!, with interval zeroth coefficient.

    Each derivative sweep seeds one normalized parameter direction. Constants
    may span the entire box; coefficient bounds therefore hold everywhere in it.
    """
    order = 4

    def __init__(self, value=0, derivative=0):
        self.c = [arb(value), arb(derivative)]+[arb(0) for _ in range(3)]

    @classmethod
    def coefficients(cls, values):
        result = cls()
        result.c = list(values)
        return result

    @staticmethod
    def cast(x):
        return x if isinstance(x, Jet) else Jet(x)

    def __add__(self, other):
        other = self.cast(other)
        return self.coefficients(a+b for a, b in zip(self.c, other.c))

    __radd__ = __add__

    def __neg__(self):
        return self.coefficients(-x for x in self.c)

    def __sub__(self, other):
        return self+-self.cast(other)

    def __rsub__(self, other):
        return self.cast(other)+-self

    def __mul__(self, other):
        other = self.cast(other)
        return self.coefficients(sum((self.c[k]*other.c[n-k] for k in range(n+1)), arb(0))
                                 for n in range(5))

    __rmul__ = __mul__

    def reciprocal(self):
        result = [1/self.c[0]]
        for n in range(1, 5):
            result.append(-sum((self.c[k]*result[n-k] for k in range(1, n+1)), arb(0))/self.c[0])
        return self.coefficients(result)

    def __truediv__(self, other):
        return self*self.cast(other).reciprocal()

    def __rtruediv__(self, other):
        return self.cast(other)*self.reciprocal()

    def __pow__(self, power):
        if not isinstance(power, int) or power < 0:
            raise ValueError("Jets support only nonnegative integer powers")
        result, base = Jet(1), self
        while power:
            if power % 2:
                result = result*base
            base = square(base)
            power //= 2
        return result

    def exp(self):
        result = [self.c[0].exp()]
        for n in range(1, 5):
            result.append(sum((k*self.c[k]*result[n-k] for k in range(1, n+1)), arb(0))/n)
        return self.coefficients(result)

    def log(self):
        inverse = self.reciprocal()
        return self.coefficients([self.c[0].log()]+[
            sum((k*self.c[k]*inverse.c[n-k] for k in range(1, n+1)), arb(0))/n
            for n in range(1, 5)])

    def sqrt(self):
        result = [self.c[0].sqrt()]
        for n in range(1, 5):
            result.append((self.c[n]-sum((result[k]*result[n-k] for k in range(1, n)), arb(0)))
                          /(2*result[0]))
        return self.coefficients(result)

    def erf(self):
        exponential = (-square(self)).exp()
        factor = 2/arb.pi().sqrt()
        return self.coefficients([self.c[0].erf()]+[
            factor*sum((k*self.c[k]*exponential.c[n-k] for k in range(1, n+1)), arb(0))/n
            for n in range(1, 5)])

    def is_finite(self):
        return all(x.is_finite() for x in self.c)
