from ._native import lib, ffi
from ._bridge import rustcall, special_errors


lib.pyfrac_init()


special_errors.update({
})


def repeated(n: 'fractions.Fraction', base, min_exp):
    p = n.numerator
    p = p.to_bytes((p.bit_length() + 7) // 8, 'big')
    q = n.denominator
    q = q.to_bytes((q.bit_length() + 7) // 8, 'big')
    result_ptr = rustcall(lib.pyfrac_repeated, p, len(p), q, len(q), base, min_exp)
    result = ffi.string(result_ptr).decode('utf-8', 'replace')
    lib.pyfrac_free(result_ptr)
    return result
