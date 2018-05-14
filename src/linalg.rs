use ::*;

pub trait VectorOps<T> where T: Copy {
  fn matrix_vector_mult(&mut self,
      w: MemArrayView2d<T>,
      x: MemArrayView1d<T>);
  /*fn transpose_matrix_vector_mult(&mut self,
      w: MemArrayView2d<T>,
      x: MemArrayView1d<T>);*/
}

impl VectorOps<f32> for MemArrayViewMut1d<f32> {
  fn matrix_vector_mult(&mut self,
      w: MemArrayView2d<f32>,
      x: MemArrayView1d<f32>)
  {
    // TODO
    assert_eq!(w.size()[0], self.size());
    assert_eq!(w.size()[1], x.size());
    assert_eq!(w.stride()[0], 1);
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    unsafe { cblas_sgemv(
        CBLAS_TRANSPOSE_CblasNoTranspose,
        sz2int(w.size()[0]),
        sz2int(w.size()[1]),
        &alpha,
        w.as_ptr(), sz2int(w.stride()[1]),
        x.as_ptr(), sz2int(x.stride()),
        &beta,
        self.as_mut_ptr(), sz2int(self.stride()),
    ) };
  }
}

impl VectorOps<f64> for MemArrayViewMut1d<f64> {
  fn matrix_vector_mult(&mut self,
      w: MemArrayView2d<f64>,
      x: MemArrayView1d<f64>)
  {
    // TODO
    assert_eq!(w.size()[0], self.size());
    assert_eq!(w.size()[1], x.size());
    assert_eq!(w.stride()[0], 1);
    let alpha: f64 = 1.0;
    let beta: f64 = 0.0;
    unsafe { cblas_dgemv(
        CBLAS_TRANSPOSE_CblasNoTranspose,
        sz2int(w.size()[0]),
        sz2int(w.size()[1]),
        &alpha,
        w.as_ptr(), sz2int(w.stride()[1]),
        x.as_ptr(), sz2int(x.stride()),
        &beta,
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

/*impl MatrixOps<f32> for MemArrayViewMut2d<f32> {
  fn matrix_mult(&mut self,
      w: MemArrayView2d<T>,
      x: MemArrayView2d<T>)
  {
  }

  fn left_transpose_matrix_mult(&mut self,
      w: MemArrayView2d<T>,
      x: MemArrayView2d<T>)
  {
  }

  fn right_transpose_matrix_mult(&mut self,
      w: MemArrayView2d<T>,
      x: MemArrayView2d<T>)
  {
  }
}*/
