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
extern crate libc;

use arrayidx::*;

use std::mem::{size_of};
use std::slice::{from_raw_parts, from_raw_parts_mut};
use std::sync::{Arc, Mutex, MutexGuard};

pub trait Mem<T> {
  fn as_slice(&self) -> &[T];
  fn as_mut_slice(&mut self) -> &mut [T];
}

pub struct AllocMem<T> {
  ptr:  *mut T,
  len:  usize,
  psz:  usize,
}

impl<T> Mem<T> for AllocMem<T> {
  fn as_slice(&self) -> &[T] {
    unsafe { from_raw_parts(self.ptr as *const _, self.len) }
  }

  fn as_mut_slice(&mut self) -> &mut [T] {
    unsafe { from_raw_parts_mut(self.ptr, self.len) }
  }
}

impl<T> AllocMem<T> {
  pub unsafe fn alloc(len: usize) -> Self {
    let psz = len * size_of::<T>();
    let ptr = unsafe { libc::malloc(psz) } as *mut T;
    AllocMem{
      ptr,
      len,
      psz,
    }
  }
}

pub struct SharedMemArray<Idx, T> where T: Copy {
  size:     Idx,
  offset:   Idx,
  stride:   Idx,
  mem:      Arc<Mutex<Mem<T>>>,
}

pub type SharedMemScalar<T>  = SharedMemArray<Index0d, T>;
pub type SharedMemArray1d<T> = SharedMemArray<Index1d, T>;
pub type SharedMemArray2d<T> = SharedMemArray<Index2d, T>;
pub type SharedMemArray3d<T> = SharedMemArray<Index3d, T>;
pub type SharedMemArray4d<T> = SharedMemArray<Index4d, T>;
pub type SharedMemArray5d<T> = SharedMemArray<Index5d, T>;

pub trait Array {
  type Idx;

  fn size(&self) -> Self::Idx;
}

pub trait MemArrayZeros: Array {
  fn zeros(size: Self::Idx) -> Self where Self: Sized;
}

pub trait MemBatchArrayZeros: Array {
  fn zeros(size: Self::Idx) -> Self where Self: Sized;
}

impl<Idx, T> Array for SharedMemArray<Idx, T> where Idx: ArrayIndex + Copy, T: Copy + 'static {
  type Idx = Idx;

  fn size(&self) -> Idx {
    self.size
  }
}

impl<Idx, T> MemArrayZeros for SharedMemArray<Idx, T> where Idx: ArrayIndex + Copy, T: Copy + 'static {
  fn zeros(size: Idx) -> Self {
    SharedMemArray{
      size:     size,
      offset:   Idx::zero(),
      stride:   size.to_packed_stride(),
      mem:      Arc::new(Mutex::new(unsafe { AllocMem::<T>::alloc(size.flat_len()) })),
    }
  }
}
