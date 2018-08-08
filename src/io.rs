/*
Copyright 2017-2018 Peter Jin

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

use ::{Mem, ZeroBits, MemArray};

use arrayidx::{ArrayIndex};
use byteorder::*;

use std::io::{Read, Write};
use std::str::{from_utf8};

pub trait NpyArrayIo<Idx, T> {
  fn deserialize<R: Read + ?Sized>(reader: &mut R) -> Result<Self, ()> where Self: Sized;
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum NpyEndianness {
  Little,
  Big,
}

impl NpyEndianness {
  #[cfg(target_endian = "little")]
  pub fn native() -> Self {
    NpyEndianness::Little
  }

  #[cfg(target_endian = "big")]
  pub fn native() -> Self {
    NpyEndianness::Big
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum NpyDtype {
  Float32,
  Float64,
  Int8,
  Int16,
  Int32,
  Int64,
  UInt8,
  UInt16,
  UInt32,
  UInt64,
}

pub trait ToNpyDtypeDesc {
  fn to_npy_dtype_desc() -> NpyDtypeDesc;
}

impl ToNpyDtypeDesc for u8 {
  fn to_npy_dtype_desc() -> NpyDtypeDesc {
    NpyDtypeDesc{
      endian:   None,
      dtype:    NpyDtype::UInt8,
    }
  }
}

impl ToNpyDtypeDesc for f32 {
  fn to_npy_dtype_desc() -> NpyDtypeDesc {
    NpyDtypeDesc{
      endian:   Some(NpyEndianness::native()),
      dtype:    NpyDtype::Float32,
    }
  }
}

impl ToNpyDtypeDesc for f64 {
  fn to_npy_dtype_desc() -> NpyDtypeDesc {
    NpyDtypeDesc{
      endian:   Some(NpyEndianness::native()),
      dtype:    NpyDtype::Float64,
    }
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct NpyDtypeDesc {
  pub endian:   Option<NpyEndianness>,
  pub dtype:    NpyDtype,
}

impl NpyDtypeDesc {
  pub fn parse(desc: &str) -> Result<Self, ()> {
    let (endian, dtype) = match desc {
      "'|u1'," => (None,                        NpyDtype::UInt8),
      "'<f4'," => (Some(NpyEndianness::Little), NpyDtype::Float32),
      "'<f8'," => (Some(NpyEndianness::Little), NpyDtype::Float64),
      _ => unimplemented!("NpyDtypeDesc: unhandled str: {}", desc),
    };
    Ok(NpyDtypeDesc{endian, dtype})
  }

  pub fn matches<T: ToNpyDtypeDesc>(&self) -> bool {
    *self == T::to_npy_dtype_desc()
  }
}

pub struct NpyHeader {
  pub dtype_desc:   NpyDtypeDesc,
  pub col_major:    bool,
  pub nd_size:      Vec<usize>,
  pub data_offset:  usize,
}

pub fn read_npy_header<R: Read + ?Sized>(reader: &mut R) -> Result<NpyHeader, ()> {
  let mut magicnum = Vec::with_capacity(6);
  for _ in 0 .. 6 {
    magicnum.push(0);
  }
  match reader.read_exact(&mut magicnum) {
    Err(_) => panic!("read_npy_header: failed to verify magicnum"),
    Ok(_) => {}
  }
  if &magicnum != b"\x93NUMPY" {
    return Err(());
  }
  //println!("DEBUG: read_npy_header: parse version bytes...");
  let major_ver = reader.read_u8().unwrap();
  let minor_ver = reader.read_u8().unwrap();
  if (major_ver, minor_ver) != (1, 0) {
    return Err(());
  }
  //println!("DEBUG: read_npy_header: parse header len...");
  let header_len = reader.read_u16::<LittleEndian>().unwrap() as usize;
  let data_offset = 10 + header_len;
  assert_eq!(data_offset % 64, 0);
  let mut header = Vec::with_capacity(header_len);
  for _ in 0 .. header_len {
    header.push(0);
  }
  //println!("DEBUG: read_npy_header: read header...");
  reader.read_exact(&mut header).unwrap();
  let header_toks: Vec<_> = from_utf8(&header).unwrap().split_whitespace().collect();
  //println!("DEBUG: read_npy_header: got header toks: {:?}", &header_toks);
  assert_eq!(header_toks[0], "{'descr':");
  assert_eq!(header_toks[2], "'fortran_order':");
  assert_eq!(header_toks[4], "'shape':");
  let dtype_desc = NpyDtypeDesc::parse(header_toks[1]).unwrap();
  let col_major: bool = header_toks[3].replace(",", "").to_lowercase().parse().unwrap();
  let mut nd_size = vec![];
  for shape_tok in header_toks[5 .. ].iter() {
    if &"}" == shape_tok {
      break;
    }
    let shape_tok = shape_tok.replace("(", "").replace(")", "").replace(",", "");
    let d: usize = shape_tok.parse().unwrap();
    nd_size.push(d);
  }
  if !col_major {
    nd_size.reverse();
  }
  //println!("DEBUG: read_npy_header: got size: {:?}", &nd_size);
  Ok(NpyHeader{
    dtype_desc,
    col_major,
    nd_size,
    data_offset,
  })
}

pub fn write_npy_header<W: Write + ?Sized>(header: &NpyHeader, writer: &mut W) -> Result<(), ()> {
  // TODO
  unimplemented!();
}

impl<Idx, T> NpyArrayIo<Idx, T> for MemArray<Idx, T> where Idx: ArrayIndex, T: ToNpyDtypeDesc + ZeroBits {
  fn deserialize<R: Read + ?Sized>(reader: &mut R) -> Result<Self, ()> {
    let header = {
      match read_npy_header(reader) {
        Err(_) => panic!(),
        Ok(header) => header,
      }
    };
    assert!(header.dtype_desc.matches::<T>());
    let size = <Idx as ArrayIndex>::from_nd(header.nd_size);
    let mut arr = MemArray::zeros(size);
    match reader.read_exact(arr.memory_mut().as_mut_bytes()) {
      Err(_) => panic!(),
      Ok(_) => {}
    }
    Ok(arr)
  }
}
