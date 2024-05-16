use std::{marker::PhantomData, sync::Arc};

use burn_compute::client::ComputeClient;
use burn_fusion::stream::Context;

use crate::{
    codegen::{calculate_num_elems_dyn_rank, CompilationInfo, CompilationSettings},
    fusion::{
        kernel::{FusionKernel, FusionKernelFactory, OutputRuntimeInfo},
        tracing::Trace,
        JitFusionHandle,
    },
    gpu::WorkgroupSize,
    kernel::elemwise_workgroup,
    Runtime,
};

#[derive(new)]
pub struct GraphOptimization<R: Runtime> {
    pub(super) trace: Trace,
    pub(super) num_operations: usize,
    pub(super) device: R::Device,
    factory: GraphKernelFactory<R>,
}

impl<R: Runtime> GraphOptimization<R> {
    pub(crate) fn execute(&mut self, context: &mut Context<'_, JitFusionHandle<R>>) {
        let client = R::client(&self.device);

        self.run_kernel(context, client, 0)
    }

    fn run_kernel(
        &mut self,
        context: &mut Context<'_, JitFusionHandle<R>>,
        client: ComputeClient<R::Server, R::Channel>,
        fastest_set_index: usize,
    ) {
        let info = self.trace.running();
        let kernel_set = &self.factory;

        let kernel = FusionKernel::create(
            kernel_set,
            &info,
            context,
            self.device.clone(),
            client,
            true,
        );

        kernel.execute();
    }
    pub(crate) fn len(&self) -> usize {
        self.num_operations
    }
}

impl<R: Runtime> FusionKernelFactory<R> for GraphKernelFactory<R> {
    fn create(
        &self,
        handles_inputs: &[JitFusionHandle<R>],
        inputs: &[&burn_tensor::repr::TensorDescription],
        outputs: &[&burn_tensor::repr::TensorDescription],
        stateful: bool, // Should be set to false when running autotune.
    ) -> crate::fusion::kernel::FusionKernel<R> {
        let workgroup_size_x = self.grid.x;
        let workgroup_size_y = self.grid.y;
        let workgroup_size = workgroup_size_x as usize;
        for h in handles_inputs {
            println!("h {:?}", h);
        }

        for o in self.info.scope.operations.iter() {
            println!("O {o:?}");
        }

        let settings = CompilationSettings::default();
        let factor = 1;

        let reference_tensor = outputs[0];
        let num_elems = calculate_num_elems_dyn_rank(&reference_tensor.shape);
        let workgroup = elemwise_workgroup(num_elems / factor, workgroup_size);
        let output_infos = outputs.iter().enumerate().map(|(pos, tensor)| {
            let size = calculate_num_elems_dyn_rank(&tensor.shape)
                * self.info.outputs[pos].elem_size::<R>();
            OutputRuntimeInfo::Array { size }
        });

        FusionKernel::new(
            self.id.clone(),
            self.info.clone(),
            settings,
            output_infos.collect(),
            workgroup,
        )
    }
}

#[derive(new)]
pub struct GraphKernelFactory<R: Runtime> {
    id: String,
    info: Arc<CompilationInfo>,
    grid: WorkgroupSize,
    _runtime: PhantomData<R>,
}
