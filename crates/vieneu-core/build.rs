//! ONNX Runtime's static CoreML objects call `__isPlatformVersionAtLeast`, which
//! lives in compiler-rt; rust-lld does not pull that in on its own.

use std::path::Path;
use std::process::Command;

fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }
    let output = Command::new("clang")
        .arg("--print-file-name=libclang_rt.osx.a")
        .output()
        .expect("clang is required to link on macOS");
    let path = String::from_utf8(output.stdout).expect("clang prints a UTF-8 path");
    let dir = Path::new(path.trim()).parent().expect("compiler-rt path has a parent");
    println!("cargo:rustc-link-search=native={}", dir.display());
    println!("cargo:rustc-link-lib=static=clang_rt.osx");
}
