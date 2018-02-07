# From https://youtu.be/zmtHaZG7pPc?t=22m29s
from ._native import lib as _lib, ffi as _ffi


special_errors = {
}

error_type = 'struct pyfrac_error *'
error_message_free = _lib.pyfrac_free


def rustcall(func, *args):
    err = _ffi.new(error_type)
    rv = func(*(args + (err,)))
    if not err[0].failed:
        return rv
    try:
        exc_class = special_errors.get(err[0].code, Exception)
        exc = exc_class(_ffi.string(err[0].message).decode('utf-8', 'replace'))
    finally:
        error_message_free(err[0].message)
    raise exc
