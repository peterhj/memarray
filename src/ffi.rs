#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]
#![allow(non_snake_case)]

#[cfg(feature = "mkl")]
pub mod mkl {
include!(concat!(env!("OUT_DIR"), "/mkl_bind.rs"));
}
