use std::{panic, mem};
use std::os::raw::c_uint;
use err::{Error, ErrorKind, Result};

#[repr(C)]
pub struct CError {
    message: *mut u8,
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
unsafe fn set_err(err: Error, err_out: *mut CError) {
    if err_out.is_null() {
        return;
    }
    let s = match err.kind {
        ErrorKind::Internal => {
            match &PANIC_INFO {
                &Some(ref s) => format!("{}\x00", s),
                &None => "no panic info\x00".to_owned(),
            }
        },
        _ => format!("{}\x00", err),
    };
    (*err_out).message = Box::into_raw(s.into_boxed_str()) as *mut u8;
    (*err_out).code = err.get_error_code();
    (*err_out).failed = 1;
}
// End from

// From https://youtu.be/zmtHaZG7pPc?t=21m54s
pub unsafe fn landingpad<F: FnOnce() -> Result<T> + panic::UnwindSafe, T>(
    f: F, err_out: *mut CError) -> T
{
    if let Ok(rv) = panic::catch_unwind(f) {
        rv.map_err(|err| set_err(err, err_out)).unwrap_or(mem::zeroed())
    } else {
        set_err(ErrorKind::Internal.into(), err_out);
        mem::zeroed()
    }
}
