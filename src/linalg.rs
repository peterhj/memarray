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

use ::*;
#[cfg(feature = "mkl")] use ::ffi::mkl::*;

#[inline]
fn sz2int(sz: usize) -> i32 {
  assert!(sz <= i32::max_value() as _);
  sz as _
}

pub trait VectorOps<T> where T: Copy {
  fn matrix_vector_mult(&mut self,
      w: MemArrayView2d<T>,
      x: MemArrayView1d<T>);
  fn transpose_matrix_vector_mult(&mut self,
      w: MemArrayView2d<T>,
      x: MemArrayView1d<T>);
}

#[cfg(feature = "mkl")]
impl<'a> VectorOps<f32> for MemArrayViewMut1d<'a, f32> {
  fn matrix_vector_mult(&mut self,
      w: MemArrayView2d<f32>,
      x: MemArrayView1d<f32>)
  {
    assert_eq!(w.size()[0], self.size());
    assert_eq!(w.size()[1], x.size());
    assert_eq!(w.stride()[0], 1);
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    unsafe { cblas_sgemv(
        CBLAS_LAYOUT_CblasColMajor,
        CBLAS_TRANSPOSE_CblasNoTrans,
        sz2int(w.size()[0]),
        sz2int(w.size()[1]),
        alpha,
        w.as_ptr(), sz2int(w.stride()[1]),
        x.as_ptr(), sz2int(x.stride()),
        beta,
        self.as_mut_ptr(), sz2int(self.stride()),
    ) };
  }

  fn transpose_matrix_vector_mult(&mut self,
      w: MemArrayView2d<f32>,
      x: MemArrayView1d<f32>)
  {
    assert_eq!(w.size()[0], x.size());
    assert_eq!(w.size()[1], self.size());
    assert_eq!(w.stride()[0], 1);
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    unsafe { cblas_sgemv(
        CBLAS_LAYOUT_CblasColMajor,
        CBLAS_TRANSPOSE_CblasTrans,
        sz2int(w.size()[0]),
        sz2int(w.size()[1]),
        alpha,
        w.as_ptr(), sz2int(w.stride()[1]),
        x.as_ptr(), sz2int(x.stride()),
        beta,
        self.as_mut_ptr(), sz2int(self.stride()),
    ) };
  }
}

#[cfg(feature = "mkl")]
impl<'a> VectorOps<f64> for MemArrayViewMut1d<'a, f64> {
  fn matrix_vector_mult(&mut self,
      w: MemArrayView2d<f64>,
      x: MemArrayView1d<f64>)
  {
    assert_eq!(w.size()[0], self.size());
    assert_eq!(w.size()[1], x.size());
    assert_eq!(w.stride()[0], 1);
    let alpha: f64 = 1.0;
    let beta: f64 = 0.0;
    unsafe { cblas_dgemv(
        CBLAS_LAYOUT_CblasColMajor,
        CBLAS_TRANSPOSE_CblasNoTrans,
        sz2int(w.size()[0]),
        sz2int(w.size()[1]),
        alpha,
        w.as_ptr(), sz2int(w.stride()[1]),
        x.as_ptr(), sz2int(x.stride()),
        beta,
        self.as_mut_ptr(), sz2int(self.stride()),
    ) };
  }

  fn transpose_matrix_vector_mult(&mut self,
      w: MemArrayView2d<f64>,
      x: MemArrayView1d<f64>)
  {
    assert_eq!(w.size()[0], x.size());
    assert_eq!(w.size()[1], self.size());
    assert_eq!(w.stride()[0], 1);
    let alpha: f64 = 1.0;
    let beta: f64 = 0.0;
    unsafe { cblas_dgemv(
        CBLAS_LAYOUT_CblasColMajor,
        CBLAS_TRANSPOSE_CblasTrans,
        sz2int(w.size()[0]),
        sz2int(w.size()[1]),
        alpha,
        w.as_ptr(), sz2int(w.stride()[1]),
        x.as_ptr(), sz2int(x.stride()),
        beta,
        self.as_mut_ptr(), sz2int(self.stride()),
    ) };
  }
}

pub trait MatrixOps<T> where T: Copy {
  fn matrix_mult(&mut self,
      w: MemArrayView2d<T>,
      x: MemArrayView2d<T>);
  fn left_transpose_matrix_mult(&mut self,
      w: MemArrayView2d<T>,
      x: MemArrayView2d<T>);
  fn right_transpose_matrix_mult(&mut self,
      w: MemArrayView2d<T>,
      x: MemArrayView2d<T>);
}

#[cfg(feature = "mkl")]
impl<'a> MatrixOps<f32> for MemArrayViewMut2d<'a, f32> {
  fn matrix_mult(&mut self,
      w: MemArrayView2d<f32>,
      x: MemArrayView2d<f32>)
  {
    assert_eq!(w.size()[0], self.size()[0]);
    assert_eq!(w.size()[1], x.size()[0]);
    assert_eq!(x.size()[1], self.size()[1]);
    assert_eq!(w.stride()[0], 1);
    assert_eq!(x.stride()[0], 1);
    assert_eq!(self.stride()[0], 1);
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    unsafe { cblas_sgemm(
        CBLAS_LAYOUT_CblasColMajor,
        CBLAS_TRANSPOSE_CblasNoTrans,
        CBLAS_TRANSPOSE_CblasNoTrans,
        sz2int(self.size()[0]),
        sz2int(self.size()[1]),
        sz2int(w.size()[1]),
        alpha,
        w.as_ptr(), sz2int(w.stride()[1]),
        x.as_ptr(), sz2int(x.stride()[1]),
        beta,
        self.as_mut_ptr(), sz2int(self.stride()[1]),
    ) };
  }

  fn left_transpose_matrix_mult(&mut self,
      w: MemArrayView2d<f32>,
      x: MemArrayView2d<f32>)
  {
    // TODO
    unimplemented!();
  }

  fn right_transpose_matrix_mult(&mut self,
      w: MemArrayView2d<f32>,
      x: MemArrayView2d<f32>)
  {
    // TODO
    unimplemented!();
  }
}

#[cfg(feature = "mkl")]
impl<'a> MatrixOps<f64> for MemArrayViewMut2d<'a, f64> {
  fn matrix_mult(&mut self,
      w: MemArrayView2d<f64>,
      x: MemArrayView2d<f64>)
  {
    assert_eq!(w.size()[0], self.size()[0]);
    assert_eq!(w.size()[1], x.size()[0]);
    assert_eq!(x.size()[1], self.size()[1]);
    assert_eq!(w.stride()[0], 1);
    assert_eq!(x.stride()[0], 1);
    assert_eq!(self.stride()[0], 1);
    let alpha: f64 = 1.0;
    let beta: f64 = 0.0;
    unsafe { cblas_dgemm(
        CBLAS_LAYOUT_CblasColMajor,
        CBLAS_TRANSPOSE_CblasNoTrans,
        CBLAS_TRANSPOSE_CblasNoTrans,
        sz2int(self.size()[0]),
        sz2int(self.size()[1]),
        sz2int(w.size()[1]),
        alpha,
        w.as_ptr(), sz2int(w.stride()[1]),
        x.as_ptr(), sz2int(x.stride()[1]),
        beta,
        self.as_mut_ptr(), sz2int(self.stride()[1]),
    ) };
  }

  fn left_transpose_matrix_mult(&mut self,
      w: MemArrayView2d<f64>,
      x: MemArrayView2d<f64>)
  {
    // TODO
    unimplemented!();
  }

  fn right_transpose_matrix_mult(&mut self,
      w: MemArrayView2d<f64>,
      x: MemArrayView2d<f64>)
  {
    // TODO
    unimplemented!();
  }
}
