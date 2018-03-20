/*
Copyright 2017 the memarray authors

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

use byteorder::*;

use std::collections::{BTreeMap};
use std::io::{Read, Write};
use std::str::{from_utf8};

pub trait NpySerialize {
  fn deserialize(reader: &mut Read) -> Self;
  fn serialize(&self, writer: &mut Write);
}

pub struct NpyArray {
}

impl NpyArray {
  pub fn flat_len(&self) -> usize {
    // TODO
    unimplemented!();
  }
}

impl NpySerialize for NpyArray {
  fn deserialize(reader: &mut Read) -> Self {
    // TODO
    unimplemented!();
  }

  fn serialize(&self, writer: &mut Write) {
    // TODO
    unimplemented!();
  }
}

pub enum NpyEndianness {
  Little,
  Big,
}

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

pub struct NpyHeader {
  dtype_desc:   (NpyEndianness, NpyDtype),
  col_major:    bool,
  c_shape:      Vec<usize>,
  data_offset:  usize,
}

pub fn read_npy_header<R>(reader: &mut R) -> Result<NpyHeader, ()> where R: Read {
  let mut magicnum = Vec::with_capacity(6);
  for _ in 0 .. 6 {
    magicnum.push(0);
  }
  reader.read_exact(&mut magicnum).unwrap();
  if &magicnum != b"\x93NUMPY" {
    return Err(());
  }
  let major_ver = reader.read_u8().unwrap();
  let minor_ver = reader.read_u8().unwrap();
  if (major_ver, minor_ver) != (1, 0) {
    return Err(());
  }
  let header_len = reader.read_u16::<LittleEndian>().unwrap() as usize;
  let mut header = Vec::with_capacity(header_len);
  for _ in 0 .. header_len {
    header.push(0);
  }
  reader.read_exact(&mut header).unwrap();
  assert_eq!((10 + header_len) % 16, 0);
  let header_toks: Vec<_> = from_utf8(&header).unwrap().split_whitespace().collect();
  assert_eq!(header_toks[0], "{'descr':");
  assert_eq!(header_toks[2], "'fortran_order':");
  assert_eq!(header_toks[4], "'shape':");
  // TODO
  unimplemented!();
}

pub fn write_npy_header<W>(header: &NpyHeader, writer: &mut W) -> Result<(), ()> where W: Write {
  // TODO
  unimplemented!();
}

pub struct NkvArchive {
  kvs:  BTreeMap<String, NpyArray>,
}

pub struct NkvHeader {
  hlen: u32,
  kvs:  BTreeMap<String, (u64, u64)>,
}

pub fn write_nkv_header<W>(archive: &NkvArchive, writer: &mut W) -> Result<NkvHeader, ()> where W: Write {
  let magicnum = b"\x93NUMKV";
  writer.write_all(magicnum).unwrap();
  let (major_ver, minor_ver) = (2, 0);
  writer.write_u8(major_ver).unwrap();
  writer.write_u8(minor_ver).unwrap();
  let mut unpadded_header_len = 4;
  for (key, _) in archive.kvs.iter() {
    let key_len = key.as_bytes().len();
    unpadded_header_len += key_len + 20;
  }
  let header_len = (12 + unpadded_header_len + 16 - 1) / 16 * 16 - 12;
  writer.write_u32::<LittleEndian>(header_len as _).unwrap();
  writer.write_u32::<LittleEndian>(archive.kvs.len() as _).unwrap();
  let mut offset = 12 + header_len;
  let mut kvoffsets = BTreeMap::new();
  for (key, ref value) in archive.kvs.iter() {
    let value_len = value.flat_len();
    writer.write_u32::<LittleEndian>(key.as_bytes().len() as _).unwrap();
    writer.write_all(key.as_bytes()).unwrap();
    writer.write_u64::<LittleEndian>(offset as _).unwrap();
    writer.write_u64::<LittleEndian>(value_len as _).unwrap();
    kvoffsets.insert(key.clone(), (offset as _, value_len as _));
    offset += (value_len + 16 - 1) / 16 * 16;
  }
  for _ in 0 .. header_len - unpadded_header_len {
    writer.write_u8(0).unwrap();
  }
  Ok(NkvHeader{
    hlen: header_len as _,
    kvs:  kvoffsets,
  })
}
