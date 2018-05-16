extern crate bindgen;

use std::env;
use std::fs;
use std::path::{PathBuf};

fn main() {
  let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

  if cfg!(feature = "mkl") {
    let mkl_dir = PathBuf::from(match env::var("MKL_HOME") {
      Ok(path) => path,
      Err(_) => match env::var("MKLROOT") {
        Ok(path) => path,
        Err(_) => "/usr".to_owned(),
      },
    });

    //println!("cargo:rustc-link-lib=mklml_gnu");
    //println!("cargo:rustc-link-lib=gomp");
    println!("cargo:rustc-link-lib=mklml_intel");
    println!("cargo:rustc-link-lib=iomp5");

    fs::remove_file(out_dir.join("mkl_bind.rs")).ok();

    let mkl_bindings = bindgen::Builder::default()
      .clang_arg(format!("-I{}", mkl_dir.join("include").as_os_str().to_str().unwrap()))
      .header("wrapped.h")
      .whitelist_type("CBLAS_LAYOUT")
      .whitelist_type("CBLAS_TRANSPOSE")
      .whitelist_type("CBLAS_UPLO")
      .whitelist_type("CBLAS_DIAG")
      .whitelist_type("CBLAS_SIDE")
      .whitelist_type("CBLAS_STORAGE")
      .whitelist_type("CBLAS_IDENTIFIER")
      .whitelist_type("CBLAS_OFFSET")
      .whitelist_type("CBLAS_ORDER")
      .whitelist_function("cblas_sdot")
      .whitelist_function("cblas_ddot")
      .whitelist_function("cblas_snrm2")
      .whitelist_function("cblas_dnrm2")
      .whitelist_function("cblas_saxpy")
      .whitelist_function("cblas_daxpy")
      .whitelist_function("cblas_sscal")
      .whitelist_function("cblas_dscal")
      .whitelist_function("cblas_sgemv")
      .whitelist_function("cblas_dgemv")
      .whitelist_function("cblas_sgemm")
      .whitelist_function("cblas_dgemm")
      .generate()
      .expect("bindgen failed to generate mkl bindings");
    mkl_bindings
      .write_to_file(out_dir.join("mkl_bind.rs"))
      .expect("bindgen failed to write mkl bindings");
  }
}
