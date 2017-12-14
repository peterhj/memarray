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

extern crate arrayidx;
extern crate byteorder;
extern crate libc;

use arrayidx::*;

use std::cell::{RefCell, Ref, RefMut};
use std::mem::{size_of};
use std::rc::{Rc};
use std::slice::{from_raw_parts, from_raw_parts_mut};
//use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

pub mod serial;

pub trait Mem<T> {
  unsafe fn ptr(&self) -> &*const T;
  unsafe fn shared_mut_ptr(&self) -> &*mut T;
  unsafe fn mut_ptr(&mut self) -> &mut *mut T;
  fn len(&self) -> usize;
}

pub struct AllocMem<T> {
  ptr:  *mut T,
  ptrc: *const T,
  len:  usize,
  psz:  usize,
}

impl<T> Mem<T> for AllocMem<T> {
  unsafe fn ptr(&self) -> &*const T {
    &self.ptrc
  }

  unsafe fn shared_mut_ptr(&self) -> &*mut T {
    &self.ptr
  }

  unsafe fn mut_ptr(&mut self) -> &mut *mut T {
    &mut self.ptr
  }

  fn len(&self) -> usize {
    self.len
  }
}

impl<T> AllocMem<T> {
  pub unsafe fn alloc(len: usize) -> Self {
    let psz = len * size_of::<T>();
    let ptr = unsafe { libc::malloc(psz) } as *mut T;
    AllocMem{
      ptr,
      ptrc: ptr,
      len,
      psz,
    }
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

impl ZeroBits for f32 {}
impl ZeroBits for f64 {}

pub trait Array {
  type Idx;

  fn size(&self) -> Self::Idx;
}

pub trait MemArrayZeros: Array {
  fn zeros(size: Self::Idx) -> Self where Self: Sized;
}

pub trait MemBatchArrayZeros: Array {
  fn zeros(size: Self::Idx, batch_size: usize) -> Self where Self: Sized;
}

pub struct SharedMemArray<Idx, T> where T: Copy {
  size:     Idx,
  offset:   Idx,
  stride:   Idx,
  //mem:      Arc<RwLock<Mem<T>>>,
  mem:      Rc<RefCell<Mem<T>>>,
}

pub type SharedMemScalar<T>  = SharedMemArray<Index0d, T>;
pub type SharedMemArray1d<T> = SharedMemArray<Index1d, T>;
pub type SharedMemArray2d<T> = SharedMemArray<Index2d, T>;
pub type SharedMemArray3d<T> = SharedMemArray<Index3d, T>;
pub type SharedMemArray4d<T> = SharedMemArray<Index4d, T>;
pub type SharedMemArray5d<T> = SharedMemArray<Index5d, T>;

impl<Idx, T> Array for SharedMemArray<Idx, T> where Idx: ArrayIndex + Copy, T: Copy + 'static {
  type Idx = Idx;

  fn size(&self) -> Idx {
    self.size
  }
}

impl<Idx, T> MemArrayZeros for SharedMemArray<Idx, T> where Idx: ArrayIndex + Copy, T: ZeroBits + Copy + 'static {
  fn zeros(size: Idx) -> Self {
    let mut mem = unsafe { AllocMem::<T>::alloc(size.flat_len()) };
    unsafe { libc::memset(*mem.mut_ptr() as *mut _, 0, mem.len() * size_of::<T>()) };
    SharedMemArray{
      size:     size,
      offset:   Idx::zero(),
      stride:   size.to_packed_stride(),
      //mem:      Arc::new(RwLock::new(mem)),
      mem:      Rc::new(RefCell::new(mem)),
    }
  }
}

impl<Idx, T> SharedMemArray<Idx, T> where Idx: ArrayIndex + Copy, T: Copy + 'static {
  fn as_view(&self) -> SharedMemArrayView<Idx, T> {
    SharedMemArrayView{
      size:     self.size,
      offset:   self.offset,
      stride:   self.stride,
      mem:      self.mem.clone(),
    }
  }
}

pub struct SharedMemArrayView<Idx, T> where T: Copy + 'static {
  size:     Idx,
  offset:   Idx,
  stride:   Idx,
  mem:      Rc<RefCell<Mem<T>>>,
}

impl<Idx, T> SharedMemArrayView<Idx, T> where Idx: ArrayIndex, T: Copy + 'static {
  pub unsafe fn as_ptr(&self) -> Ref<*const T> {
    let mem = self.mem.borrow();
    Ref::map(mem, |mem| unsafe { mem.ptr() })
  }

  pub unsafe fn as_mut_ptr(&self) -> RefMut<*mut T> {
    let mem = self.mem.borrow_mut();
    RefMut::map(mem, |mem| unsafe { mem.mut_ptr() })
  }
}
