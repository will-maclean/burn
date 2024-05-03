use crate::{
    backend::{Backend, BackendBridge},
    ops::FloatTensor,
    DType,
};

pub enum MultiPrecisionFloatTensor<B: MultiPrecisionBackend, const D: usize> {
    F16(FloatTensor<B::F16Backend, D>),
    F32(FloatTensor<B::F32Backend, D>),
    F64(FloatTensor<B::F64Backend, D>),
}

pub trait MultiPrecisionBackend: Sized {
    type Bridge<O: Backend, T: Backend>: BackendBridge<O, Target = T>;

    type F32Backend: Backend;
    type F64Backend: Backend;
    type F16Backend: Backend;

    fn cast(
        tensor: MultiPrecisionFloatTensor<Self, 2>,
        dtype: DType,
    ) -> MultiPrecisionFloatTensor<Self, 2> {
        match tensor {
            MultiPrecisionFloatTensor::F16(tensor) => match dtype {
                DType::F16 => todo!(),
                DType::F32 => {
                    MultiPrecisionFloatTensor::F32(
                        <Bridge<Self, Self::F16Backend, Self::F32Backend>>::into_target(
                            tensor, None,
                        ),
                    )
                }
                DType::F64 => todo!(),
                _ => panic!("Unsupported."),
            },
            MultiPrecisionFloatTensor::F32(_) => todo!(),
            MultiPrecisionFloatTensor::F64(_) => todo!(),
        }
    }
}

type Bridge<B, O, T> = <B as MultiPrecisionBackend>::Bridge<O, T>;
