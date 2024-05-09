use crate::{fusion::tracing::Trace, Runtime};

#[derive(new)]
pub struct SubGraphOptimization<R: Runtime> {
    pub(super) trace: Trace,
    pub(super) num_operations: usize,
    pub(super) device: R::Device,
}
