use std::{panic, mem, fmt};
use std::os::raw::{c_uint, c_char};
use std::ffi::CString;

pub trait CError: fmt::Display {
    fn get_error_code(&self) -> c_uint;
}

struct PanicError();

impl fmt::Display for PanicError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match unsafe {&PANIC_INFO} {
            &Some(ref s) => write!(f, "{}", s),
            &None => write!(f, "no panic info"),
        }
    }
}

impl CError for PanicError {
    fn get_error_code(&self) -> c_uint { 0 }
}

// use err::{Error, ErrorKind, Result};

#[repr(C)]
pub struct NativeError {
    message: *mut c_char,
    failed: c_uint,
    code: c_uint,
}

static mut PANIC_INFO: Option<String> = None;

// From https://youtu.be/zmtHaZG7pPc?t=21m29s
fn silent_panic_handler(pi: &panic::PanicInfo) {
    let pl = pi.payload();
    let payload = if let Some(s) = pl.downcast_ref::<&str>() { s }
    else if let Some(s) = pl.downcast_ref::<String>() { &s }
    else { "?" };
    let position = if let Some(p) = pi.location() {
        format!("At {}:{}: ", p.file(), p.line())
    }
    else { "".to_owned() };
    unsafe {
        PANIC_INFO = Some(format!("{}{}", position, payload));
    }
}
// End from

pub fn set_panic_hook() {
    panic::set_hook(Box::new(silent_panic_handler));
}

// From https://youtu.be/zmtHaZG7pPc?t=21m39s
unsafe fn set_err(err: &CError, err_out: *mut NativeError) {
    if err_out.is_null() {
        return;
    }
    let s = CString::new(format!("{}", err)).unwrap_or_else(
        |_| CString::new("<invalid error message>").unwrap());
    (*err_out).message = s.into_raw();
    (*err_out).code = err.get_error_code();
    (*err_out).failed = 1;
}
// End from

// From https://youtu.be/zmtHaZG7pPc?t=21m54s
pub unsafe fn landingpad<F: FnOnce() -> Result<T, E> + panic::UnwindSafe, T, E: CError>(
    f: F, err_out: *mut NativeError) -> T
{
    if let Ok(rv) = panic::catch_unwind(f) {
        rv.map_err(|err| set_err(&err, err_out)).unwrap_or(mem::zeroed())
    } else {
        set_err(&PanicError(), err_out);
        mem::zeroed()
    }
}

// From https://youtu.be/zmtHaZG7pPc?t=22m09s
#[macro_use]
macro_rules! export (
    ($n:ident($($an:ident: $aty:ty),*) -> Result<$rv:ty> $body:block) => (
        #[no_mangle]
        pub unsafe extern "C" fn $n($($an: $aty,)* err: *mut NativeError) -> $rv
        {
            landingpad(|| { let e: Result<$rv> = $body; e }, err)
        }
    );
);

pub trait FromC<T> {
    unsafe fn from_c(T) -> Self;
}

impl<T, U> FromC<T> for U where U: From<T> {
    unsafe fn from_c(t: T) -> Self {
        From::from(t)
    }
}

#[allow(unused_macros)]
#[macro_use]
macro_rules! export_using_from_c (
    ($n:ident($($an:ident: $aty:ty),*) -> Result<$rv:ty> = $fn:ident) => (
        export!($n($($an: $aty),*) -> Result<$rv>
        {
            $fn($(FromC::from_c($an)),*)
        });
    );
);
