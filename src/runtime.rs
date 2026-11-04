//! Locate and initialize the ONNX Runtime shared library.
//!
//! Release bundles ship ONNX Runtime next to this library (`lib/`). Unless the
//! host sets `ORT_DYLIB_PATH`, initialization looks for a sibling runtime first
//! and only then falls back to the platform loader, which keeps existing
//! deployments working. Load failures are reported as errors instead of
//! aborting the process: `ort` panics when the dynamic library is missing, and
//! a panic escaping an `extern "C"` boundary would terminate the host.

use crate::{Error, Result};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

static RUNTIME_INIT: OnceLock<std::result::Result<(), String>> = OnceLock::new();

/// Ensure ONNX Runtime is loadable, preferring the runtime bundled next to this
/// library. Idempotent: the outcome of the first call is cached.
pub fn ensure_onnx_runtime() -> Result<()> {
    RUNTIME_INIT
        .get_or_init(initialize)
        .clone()
        .map_err(Error::Ort)
}

fn initialize() -> std::result::Result<(), String> {
    if std::env::var("ORT_DYLIB_PATH")
        .map(|value| !value.trim().is_empty())
        .unwrap_or(false)
    {
        return Ok(());
    }

    let bundled = library_dir().and_then(|dir| find_runtime_library(&dir));
    let path = bundled.unwrap_or_else(default_library_name);
    let label = path.display().to_string();

    // Only the path is configured here; creating an environment is left to the
    // host (or to `ort`'s lazy default), as recommended for library crates.
    let _ = ort::init_from(path.to_string_lossy().into_owned());
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = ort::api();
    })) {
        Ok(()) => Ok(()),
        Err(_) => Err(format!(
            "ONNX Runtime could not be loaded from {label}; keep the bundled lib/ \
             layout intact or set ORT_DYLIB_PATH"
        )),
    }
}

/// Name of the runtime shipped in the release bundles.
fn default_library_name() -> PathBuf {
    let name = if cfg!(windows) {
        "onnxruntime.dll"
    } else if cfg!(target_os = "macos") {
        "libonnxruntime.dylib"
    } else {
        "libonnxruntime.so"
    };
    PathBuf::from(name)
}

/// Find an ONNX Runtime shared library inside `dir`, preferring the canonical
/// (unversioned) file name over versioned copies such as `libonnxruntime.so.1`.
fn find_runtime_library(dir: &Path) -> Option<PathBuf> {
    let preferred = dir.join(default_library_name());
    if preferred.is_file() {
        return Some(preferred);
    }

    let mut candidates = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .map(is_runtime_filename)
                .unwrap_or(false)
                && path.is_file()
        })
        .collect::<Vec<_>>();
    candidates.sort();
    candidates.into_iter().next()
}

fn is_runtime_filename(name: &str) -> bool {
    if cfg!(windows) {
        name.starts_with("onnxruntime") && name.ends_with(".dll")
    } else if cfg!(target_os = "macos") {
        name.starts_with("libonnxruntime") && name.ends_with(".dylib")
    } else {
        name.starts_with("libonnxruntime") && (name.ends_with(".so") || name.contains(".so."))
    }
}

/// Directory containing the currently loaded copy of this library.
#[cfg(unix)]
fn library_dir() -> Option<PathBuf> {
    use std::ffi::CStr;
    use std::os::raw::{c_char, c_int, c_void};

    #[repr(C)]
    struct DlInfo {
        dli_fname: *const c_char,
        dli_fbase: *mut c_void,
        dli_sname: *const c_char,
        dli_saddr: *mut c_void,
    }

    #[cfg_attr(target_os = "linux", link(name = "dl"))]
    extern "C" {
        fn dladdr(address: *const c_void, info: *mut DlInfo) -> c_int;
    }

    extern "C" fn anchor() {}

    let mut info = DlInfo {
        dli_fname: std::ptr::null(),
        dli_fbase: std::ptr::null_mut(),
        dli_sname: std::ptr::null(),
        dli_saddr: std::ptr::null_mut(),
    };
    // SAFETY: `dladdr` only reads the address and writes into `info`.
    if unsafe { dladdr(anchor as *const c_void, &mut info) } == 0 || info.dli_fname.is_null() {
        return None;
    }
    // SAFETY: `dladdr` points `dli_fname` at a NUL-terminated path string.
    let raw = unsafe { CStr::from_ptr(info.dli_fname) }.to_str().ok()?;
    let path = std::fs::canonicalize(raw).unwrap_or_else(|_| PathBuf::from(raw));
    path.parent().map(Path::to_path_buf)
}

/// Directory containing the currently loaded copy of this library.
#[cfg(windows)]
fn library_dir() -> Option<PathBuf> {
    use std::os::windows::ffi::OsStringExt;

    type Hmodule = *mut core::ffi::c_void;

    const FROM_ADDRESS: u32 = 0x0000_0004;
    const UNCHANGED_REFCOUNT: u32 = 0x0000_0002;

    #[link(name = "kernel32")]
    extern "system" {
        fn GetModuleHandleExW(flags: u32, module_name: *const u16, module: *mut Hmodule) -> i32;
        fn GetModuleFileNameW(module: Hmodule, filename: *mut u16, size: u32) -> u32;
    }

    extern "system" fn anchor() {}

    let mut module: Hmodule = std::ptr::null_mut();
    // SAFETY: `anchor` is an address inside this module; the call neither
    // increments the reference count nor retains the address.
    let ok = unsafe {
        GetModuleHandleExW(
            FROM_ADDRESS | UNCHANGED_REFCOUNT,
            anchor as *const u16,
            &mut module,
        )
    };
    if ok == 0 || module.is_null() {
        return None;
    }

    let mut buffer = vec![0u16; 32_768];
    // SAFETY: `module` is a valid module handle and `buffer` is writable.
    let length = unsafe { GetModuleFileNameW(module, buffer.as_mut_ptr(), buffer.len() as u32) };
    if length == 0 || length as usize >= buffer.len() {
        return None;
    }
    buffer.truncate(length as usize);
    let path = PathBuf::from(std::ffi::OsString::from_wide(&buffer));
    path.parent().map(Path::to_path_buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "tabular-onnx-runtime-{name}-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn finds_canonical_runtime_next_to_library() {
        let dir = scratch("canonical");
        let runtime = dir.join(default_library_name());
        fs::write(&runtime, b"stub").unwrap();
        assert_eq!(find_runtime_library(&dir), Some(runtime));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn ignores_unrelated_files() {
        let dir = scratch("unrelated");
        fs::write(dir.join("not-a-runtime.txt"), b"stub").unwrap();
        assert_eq!(find_runtime_library(&dir), None);
        let _ = fs::remove_dir_all(&dir);
    }
}
