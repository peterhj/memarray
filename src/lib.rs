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

#![feature(allocator_api)]
#![feature(collections_range)]
#![feature(ptr_internals)]
//#![feature(specialization)]

extern crate arrayidx;
extern crate byteorder;
extern crate float;
extern crate parking_lot;

use arrayidx::*;
use float::stub::{f16_stub};
use parking_lot::{RwLock};

use std::alloc::{Alloc, Global};
use std::cell::{RefCell};
use std::marker::{PhantomData};
use std::mem::{size_of};
use std::ops::{RangeBounds};
use std::ptr::{NonNull, null_mut, write_bytes};
use std::rc::{Rc};
use std::slice::{from_raw_parts, from_raw_parts_mut};
use std::sync::{Arc};

pub mod ffi;
pub mod linalg;
pub mod serial;

fn i2idx(i: isize, len: usize) -> usize {
  let u = i as usize;
  if u < len {
    return u;
  }
  let u = (len as isize + i) as usize;
  if u < len {
    return u;
  }
  panic!();
}

pub trait Mem<T> where T: Copy {
  unsafe fn as_ptr(&self) -> *const T;
  unsafe fn as_mut_ptr(&mut self) -> *mut T;
  fn as_slice(&self) -> &[T];
  fn as_mut_slice(&mut self) -> &mut [T];
}

pub struct HeapMem<T> where T: Copy {
  buf:  *mut T,
  len:  usize,
  phsz: usize,
}

impl<T> Drop for HeapMem<T> where T: Copy {
  fn drop(&mut self) {
    assert!(!self.buf.is_null());
    let p = unsafe { NonNull::new_unchecked(self.buf) };
    self.buf = null_mut();
    match unsafe { Global::default().dealloc_array(p, self.len) } {
      Err(_) => panic!(),
      Ok(_) => {}
    }
  }
}

impl<T> HeapMem<T> where T: Copy {
  pub unsafe fn alloc(len: usize) -> Self {
    let phsz = len * size_of::<T>();
    let p = match Global::default().alloc_array(len) {
      Err(_) => panic!(),
      Ok(p) => p,
    };
    HeapMem{
      buf:  p.as_ptr(),
      len:  len,
      phsz: phsz,
    }
  }
}

impl<T> Mem<T> for HeapMem<T> where T: Copy {
  unsafe fn as_ptr(&self) -> *const T {
    self.buf
  }

  unsafe fn as_mut_ptr(&mut self) -> *mut T {
    self.buf
  }

  fn as_slice(&self) -> &[T] {
    unsafe { from_raw_parts(self.buf, self.len) }
  }

  fn as_mut_slice(&mut self) -> &mut [T] {
    unsafe { from_raw_parts_mut(self.buf, self.len) }
  }
}

pub trait ZeroBits: Copy {}

impl ZeroBits for u8 {}
impl ZeroBits for u16 {}
impl ZeroBits for u32 {}
impl ZeroBits for u64 {}
impl ZeroBits for usize {}

impl ZeroBits for i8 {}
impl ZeroBits for i16 {}
impl ZeroBits for i32 {}
impl ZeroBits for i64 {}
impl ZeroBits for isize {}

impl ZeroBits for f16_stub {}
impl ZeroBits for f32 {}
impl ZeroBits for f64 {}

pub trait Shape {
  type Shape;

  fn shape(&self) -> Self::Shape;
}

pub trait Array: Shape {
  type Idx: ArrayIndex;

  fn size(&self) -> Self::Idx;

  fn flat_size(&self) -> usize {
    self.size().flat_len()
  }
}

pub trait DenseArray: Array {
  fn offset(&self) -> Self::Idx;
  fn stride(&self) -> Self::Idx;

  fn flat_offset(&self) -> usize {
    self.offset().flat_index(&self.stride())
  }

  fn is_packed(&self) -> bool {
    self.size().is_packed(&self.stride())
  }
}

pub trait BatchArray: Array {
  fn max_batch_size(&self) -> usize;
  fn batch_size(&self) -> usize;
  fn set_batch_size(&mut self, new_batch_sz: usize);
}

pub trait ZerosShape: Shape {
  fn zeros(size: Self::Shape) -> Self where Self: Sized;
}

/*pub trait MemArrayZeros: Array {
  fn zeros(size: Self::Idx) -> Self where Self: Sized;
}

pub trait MemBatchArrayZeros: Array {
  fn zeros(size: Self::Idx, batch_size: usize) -> Self where Self: Sized;
}*/

pub struct MemArray<Idx, T, M=HeapMem<T>> where T: Copy, M: Mem<T> {
  size:     Idx,
  offset:   Idx,
  stride:   Idx,
  mem:      M,
  _mrk:     PhantomData<*mut T>,
}

pub type MemScalar<T>  = MemArray0d<T>;
pub type MemArray0d<T> = MemArray<Index0d, T>;
pub type MemArray1d<T> = MemArray<Index1d, T>;
pub type MemArray2d<T> = MemArray<Index2d, T>;
pub type MemArray3d<T> = MemArray<Index3d, T>;
pub type MemArray4d<T> = MemArray<Index4d, T>;

/*impl<Idx, T> MemArray<Idx, T> where Idx: ArrayIndex, T: ZeroBits + Copy {
  pub fn zeros(size: Idx) -> Self {
  }
}*/

impl<Idx, T> ZerosShape for MemArray<Idx, T> where Idx: ArrayIndex, T: ZeroBits + Copy {
  //type Shape = Idx;

  fn zeros(size: Self::Shape) -> Self {
    let stride = size.to_packed_stride();
    let mem = unsafe { HeapMem::<T>::alloc(size.flat_len()) };
    // The memory is uninitialized, zero it using memset.
    unsafe { write_bytes::<T>(mem.buf, 0, mem.len) };
    MemArray{
      size:     size,
      offset:   Idx::zero(),
      stride:   stride,
      mem:      mem,
      _mrk:     PhantomData,
    }
  }
}

impl<Idx, T> Shape for MemArray<Idx, T> where Idx: ArrayIndex, T: Copy {
  type Shape = Idx;

  fn shape(&self) -> Idx {
    self.size.clone()
  }
}

impl<Idx, T> Array for MemArray<Idx, T> where Idx: ArrayIndex, T: Copy {
  type Idx = Idx;

  fn size(&self) -> Idx {
    self.size.clone()
  }
}

impl<Idx, T> DenseArray for MemArray<Idx, T> where Idx: ArrayIndex, T: Copy {
  fn offset(&self) -> Idx {
    self.offset.clone()
  }

  fn stride(&self) -> Idx {
    self.stride.clone()
  }
}

impl<Idx, T> MemArray<Idx, T> where Idx: ArrayIndex, T: Copy {
  /*pub unsafe fn as_ptr(&self) -> *const T {
    self.mem.as_ptr().offset(self.flat_offset() as _)
  }

  pub unsafe fn as_mut_ptr(&self) -> *mut T {
    self.mem.as_mut_ptr().offset(self.flat_offset() as _)
  }*/

  pub fn as_view<'a>(&'a self) -> MemArrayView<'a, Idx, T> {
    MemArrayView{
      size:     self.size.clone(),
      offset:   self.offset.clone(),
      stride:   self.stride.clone(),
      mem:      &self.mem,
    }
  }

  pub fn as_view_mut<'a>(&'a mut self) -> MemArrayViewMut<'a, Idx, T> {
    MemArrayViewMut{
      size:     self.size.clone(),
      offset:   self.offset.clone(),
      stride:   self.stride.clone(),
      mem:      &mut self.mem,
    }
  }

  pub fn flat_view<'a>(&'a self) -> Option<MemArrayView1d<'a, T>> {
    if self.is_packed() {
      let flat_size = self.flat_size();
      Some(MemArrayView{
        size:       flat_size,
        offset:     0,
        stride:     flat_size.to_packed_stride(),
        mem:        &self.mem,
      })
    } else {
      None
    }
  }

  pub fn flat_view_mut<'a>(&'a mut self) -> Option<MemArrayViewMut1d<'a, T>> {
    if self.is_packed() {
      let flat_size = self.flat_size();
      Some(MemArrayViewMut{
        size:       flat_size,
        offset:     0,
        stride:     flat_size.to_packed_stride(),
        mem:        &mut self.mem,
      })
    } else {
      None
    }
  }
}

pub struct MemOuterBatchArray<Idx, T> where T: Copy {
  size:     Idx,
  offset:   Idx,
  stride:   Idx,
  batch_sz:     usize,
  max_batch_sz: usize,
  mem:      HeapMem<T>,
}

pub type MemOuterBatchArray0d<T> = MemOuterBatchArray<Index0d, T>;
pub type MemOuterBatchArray1d<T> = MemOuterBatchArray<Index1d, T>;
pub type MemOuterBatchArray2d<T> = MemOuterBatchArray<Index2d, T>;
pub type MemOuterBatchArray3d<T> = MemOuterBatchArray<Index3d, T>;
pub type MemOuterBatchArray4d<T> = MemOuterBatchArray<Index4d, T>;

impl<Idx, T> MemOuterBatchArray<Idx, T> where Idx: ArrayIndex, T: Copy {
  pub fn as_view<'a>(&'a self) -> MemArrayView<'a, <Idx as ArrayIndex>::Above, T> {
    MemArrayView{
      size:     self.size.index_append(self.batch_sz),
      offset:   self.offset.index_append(0),
      stride:   self.stride.stride_append_packed(self.batch_sz),
      mem:      &self.mem,
    }
  }

  pub fn as_view_mut<'a>(&'a mut self) -> MemArrayViewMut<'a, <Idx as ArrayIndex>::Above, T> {
    MemArrayViewMut{
      size:     self.size.index_append(self.batch_sz),
      offset:   self.offset.index_append(0),
      stride:   self.stride.stride_append_packed(self.batch_sz),
      mem:      &mut self.mem,
    }
  }
}

pub struct MemArrayView<'a, Idx, T> where /*Idx: 'static,*/ T: Copy + 'static {
  size:     Idx,
  offset:   Idx,
  stride:   Idx,
  mem:      &'a Mem<T>,
}

pub type MemArrayView0d<'a, T> = MemArrayView<'a, Index0d, T>;
pub type MemArrayView1d<'a, T> = MemArrayView<'a, Index1d, T>;
pub type MemArrayView2d<'a, T> = MemArrayView<'a, Index2d, T>;
pub type MemArrayView3d<'a, T> = MemArrayView<'a, Index3d, T>;
pub type MemArrayView4d<'a, T> = MemArrayView<'a, Index4d, T>;

impl<'a, Idx, T> Shape for MemArrayView<'a, Idx, T> where Idx: ArrayIndex, T: Copy + 'static {
  type Shape = Idx;

  fn shape(&self) -> Idx {
    self.size.clone()
  }
}

impl<'a, Idx, T> Array for MemArrayView<'a, Idx, T> where Idx: ArrayIndex, T: Copy + 'static {
  type Idx = Idx;

  fn size(&self) -> Idx {
    self.size.clone()
  }
}

impl<'a, Idx, T> DenseArray for MemArrayView<'a, Idx, T> where Idx: ArrayIndex, T: Copy + 'static {
  fn offset(&self) -> Idx {
    self.offset.clone()
  }

  fn stride(&self) -> Idx {
    self.stride.clone()
  }
}

impl<'a, Idx, T> MemArrayView<'a, Idx, T> where Idx: ArrayIndex, T: Copy + 'static {
  pub unsafe fn as_ptr(&self) -> *const T {
    self.mem.as_ptr().offset(self.flat_offset() as _)
  }

  pub fn flat_slice(&self) -> Option<&[T]> {
    if !self.is_packed() {
      return None;
    }
    Some(&self.mem.as_slice()[self.flat_offset() .. self.flat_offset() + self.flat_size()])
  }
}

impl<'a, T> MemArrayView1d<'a, T> where T: Copy + 'static {
  pub fn as_slice(&self) -> &[T] {
    self.flat_slice().unwrap()
  }

  pub fn view<R>(self, r: R) -> MemArrayView<'a, usize, T>
  where R: RangeBounds<usize>,
  {
    let (start_idx, end_idx) = range2idxs_1d(r, self.size);
    let view_size = end_idx - start_idx;
    let view_offset = self.offset + start_idx;
    MemArrayView{
      size:     view_size,
      offset:   view_offset,
      stride:   self.stride,
      mem:      self.mem,
    }
  }
}

impl<'a, T> MemArrayView2d<'a, T> where T: Copy + 'static {
  pub fn view<R0, R1>(self, r0: R0, r1: R1) -> MemArrayView<'a, [usize; 2], T>
  where R0: RangeBounds<usize>,
        R1: RangeBounds<usize>,
  {
    let (start_idx, end_idx) = range2idxs_2d(r0, r1, self.size);
    let view_size = end_idx.index_sub(&start_idx);
    let view_offset = self.offset.index_add(&start_idx);
    MemArrayView{
      size:     view_size,
      offset:   view_offset,
      stride:   self.stride,
      mem:      self.mem,
    }
  }
}

impl<'a, T> MemArrayView3d<'a, T> where T: Copy + 'static {
  pub fn view<R0, R1, R2>(self, r0: R0, r1: R1, r2: R2) -> MemArrayView<'a, [usize; 3], T>
  where R0: RangeBounds<usize>,
        R1: RangeBounds<usize>,
        R2: RangeBounds<usize>,
  {
    let (start_idx, end_idx) = range2idxs_3d(r0, r1, r2, self.size);
    let view_size = end_idx.index_sub(&start_idx);
    let view_offset = self.offset.index_add(&start_idx);
    MemArrayView{
      size:     view_size,
      offset:   view_offset,
      stride:   self.stride,
      mem:      self.mem,
    }
  }
}

impl<'a, T> MemArrayView4d<'a, T> where T: Copy + 'static {
  pub fn view<R0, R1, R2, R3>(self, r0: R0, r1: R1, r2: R2, r3: R3) -> MemArrayView<'a, [usize; 4], T>
  where R0: RangeBounds<usize>,
        R1: RangeBounds<usize>,
        R2: RangeBounds<usize>,
        R3: RangeBounds<usize>,
  {
    let (start_idx, end_idx) = range2idxs_4d(r0, r1, r2, r3, self.size);
    let view_size = end_idx.index_sub(&start_idx);
    let view_offset = self.offset.index_add(&start_idx);
    MemArrayView{
      size:     view_size,
      offset:   view_offset,
      stride:   self.stride.clone(),
      mem:      self.mem,
    }
  }

  pub fn squeeze(self, axis: isize, idx: usize) -> MemArrayView3d<'a, T> {
    let axis = i2idx(axis, 4);
    // TODO
    unimplemented!();
  }
}

pub struct MemArrayViewMut<'a, Idx, T> where T: Copy + 'static {
  size:     Idx,
  offset:   Idx,
  stride:   Idx,
  mem:      &'a mut Mem<T>,
}

pub type MemArrayViewMut0d<'a, T> = MemArrayViewMut<'a, Index0d, T>;
pub type MemArrayViewMut1d<'a, T> = MemArrayViewMut<'a, Index1d, T>;
pub type MemArrayViewMut2d<'a, T> = MemArrayViewMut<'a, Index2d, T>;
pub type MemArrayViewMut3d<'a, T> = MemArrayViewMut<'a, Index3d, T>;
pub type MemArrayViewMut4d<'a, T> = MemArrayViewMut<'a, Index4d, T>;

impl<'a, Idx, T> Shape for MemArrayViewMut<'a, Idx, T> where Idx: ArrayIndex, T: Copy + 'static {
  type Shape = Idx;

  fn shape(&self) -> Idx {
    self.size.clone()
  }
}

impl<'a, Idx, T> Array for MemArrayViewMut<'a, Idx, T> where Idx: ArrayIndex, T: Copy + 'static {
  type Idx = Idx;

  fn size(&self) -> Idx {
    self.size.clone()
  }
}

impl<'a, Idx, T> DenseArray for MemArrayViewMut<'a, Idx, T> where Idx: ArrayIndex, T: Copy + 'static {
  fn offset(&self) -> Idx {
    self.offset.clone()
  }

  fn stride(&self) -> Idx {
    self.stride.clone()
  }
}

impl<'a, Idx, T> Into<MemArrayView<'a, Idx, T>> for MemArrayViewMut<'a, Idx, T> where Idx: ArrayIndex, T: Copy + 'static {
  fn into(self) -> MemArrayView<'a, Idx, T> {
    MemArrayView{
      size:     self.size,
      offset:   self.offset,
      stride:   self.stride,
      mem:      self.mem,
    }
  }
}

impl<'a, Idx, T> MemArrayViewMut<'a, Idx, T> where Idx: ArrayIndex, T: Copy + 'static {
  pub unsafe fn as_ptr(&self) -> *const T {
    self.mem.as_ptr().offset(self.flat_offset() as _)
  }

  pub unsafe fn as_mut_ptr(&mut self) -> *mut T {
    self.mem.as_mut_ptr().offset(self.flat_offset() as _)
  }

  pub fn flat_slice(&self) -> Option<&[T]> {
    if !self.is_packed() {
      return None;
    }
    Some(&self.mem.as_slice()[self.flat_offset() .. self.flat_offset() + self.flat_size()])
  }

  pub fn flat_slice_mut(&mut self) -> Option<&mut [T]> {
    if !self.is_packed() {
      return None;
    }
    let off = self.flat_offset();
    let len = self.flat_size();
    Some(&mut self.mem.as_mut_slice()[off .. off + len])
  }
}

impl<'a, T> MemArrayViewMut1d<'a, T> where T: Copy + 'static {
  pub fn as_slice(&self) -> &[T] {
    self.flat_slice().unwrap()
  }

  pub fn as_mut_slice(&mut self) -> &mut [T] {
    self.flat_slice_mut().unwrap()
  }

  pub fn view_mut<R>(self, r: R) -> MemArrayViewMut<'a, usize, T>
  where R: RangeBounds<usize>,
  {
    let (start_idx, end_idx) = range2idxs_1d(r, self.size);
    let view_size = end_idx - start_idx;
    let view_offset = self.offset + start_idx;
    MemArrayViewMut{
      size:     view_size,
      offset:   view_offset,
      stride:   self.stride,
      mem:      self.mem,
    }
  }
}

impl<'a, T> MemArrayViewMut2d<'a, T> where T: Copy + 'static {
  pub fn view_mut<R0, R1>(self, r0: R0, r1: R1) -> MemArrayViewMut<'a, [usize; 2], T>
  where R0: RangeBounds<usize>,
        R1: RangeBounds<usize>,
  {
    let (start_idx, end_idx) = range2idxs_2d(r0, r1, self.size);
    let view_size = end_idx.index_sub(&start_idx);
    let view_offset = self.offset.index_add(&start_idx);
    MemArrayViewMut{
      size:     view_size,
      offset:   view_offset,
      stride:   self.stride,
      mem:      self.mem,
    }
  }
}

impl<'a, T> MemArrayViewMut3d<'a, T> where T: Copy + 'static {
  pub fn view_mut<R0, R1, R2>(self, r0: R0, r1: R1, r2: R2) -> MemArrayViewMut<'a, [usize; 3], T>
  where R0: RangeBounds<usize>,
        R1: RangeBounds<usize>,
        R2: RangeBounds<usize>,
  {
    let (start_idx, end_idx) = range2idxs_3d(r0, r1, r2, self.size);
    let view_size = end_idx.index_sub(&start_idx);
    let view_offset = self.offset.index_add(&start_idx);
    MemArrayViewMut{
      size:     view_size,
      offset:   view_offset,
      stride:   self.stride,
      mem:      self.mem,
    }
  }
}

impl<'a, T> MemArrayViewMut4d<'a, T> where T: Copy + 'static {
  pub fn view_mut<R0, R1, R2, R3>(self, r0: R0, r1: R1, r2: R2, r3: R3) -> MemArrayViewMut<'a, [usize; 4], T>
  where R0: RangeBounds<usize>,
        R1: RangeBounds<usize>,
        R2: RangeBounds<usize>,
        R3: RangeBounds<usize>,
  {
    let (start_idx, end_idx) = range2idxs_4d(r0, r1, r2, r3, self.size);
    let view_size = end_idx.index_sub(&start_idx);
    let view_offset = self.offset.index_add(&start_idx);
    MemArrayViewMut{
      size:     view_size,
      offset:   view_offset,
      stride:   self.stride.clone(),
      mem:      self.mem,
    }
  }

  pub fn squeeze_mut(self, axis: isize, idx: usize) -> MemArrayViewMut3d<'a, T> {
    let axis = i2idx(axis, 4);
    // TODO
    unimplemented!();
  }
}

/*pub struct RWMemArray<Idx, T> where T: Copy {
  size:     Idx,
  offset:   Idx,
  stride:   Idx,
  mem:      Rc<RefCell<HeapMem<T>>>,
}

pub struct SharedMemArray<Idx, T> where T: Copy {
  size:     Idx,
  offset:   Idx,
  stride:   Idx,
  mem:      Arc<RwLock<HeapMem<T>>>,
}*/
