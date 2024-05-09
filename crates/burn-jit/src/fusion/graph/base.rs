use crate::{
    fusion::{tracing::TraceBuilder, JitOptimization},
    gpu::{gpu, LazyProcedure, Scope, Variable, WorkgroupSize},
    Runtime,
};
use burn_fusion::{OptimizationBuilder, OptimizationProperties, OptimizationStatus};
use burn_tensor::repr::{BinaryOperationDescription, OperationDescription, TensorId};

#[derive(Clone)]
pub struct LaunchSettings {
    workgroup_size: WorkgroupSizeSettings,
    workgroup: WorkgroupSettings,
}

#[derive(Clone, PartialEq, Eq)]
pub enum WorkgroupSizeSettings {
    Any,
    Fixed(WorkgroupSize),
}

#[derive(Clone, PartialEq, Eq)]
pub enum WorkgroupSettings {
    Elemwise,
    ElemwiseShape(Vec<usize>),
    Custom,
}

impl LaunchSettings {
    pub fn is_compatible_with(&self, other: &Self) -> bool {
        match self.workgroup_size {
            WorkgroupSizeSettings::Any => match self.workgroup {
                WorkgroupSettings::Elemwise => return true,
                _ => return self.workgroup == other.workgroup,
            },
            WorkgroupSizeSettings::Fixed(_) => {
                if self.workgroup_size == other.workgroup_size {
                    match self.workgroup {
                        WorkgroupSettings::Elemwise => return true,
                        _ => return self.workgroup == other.workgroup,
                    }
                }
            }
        }

        false
    }
}

pub trait Node: Send + Sync {
    fn inputs(&self) -> &[TensorId];
    fn outputs(&self) -> &[TensorId];
    fn visit_input(&mut self, input: &TensorId);
    fn visit_output(&mut self, output: &TensorId);
    fn reset(&mut self);
    fn all_input_visited(&self) -> bool;
    fn any_output_visited(&self) -> bool;
    fn launch_settings(&self) -> LaunchSettings;
    fn register(&self, builder: &mut TraceBuilder);
}

pub trait SubGraph: Send + Sync {
    fn register(self: Box<Self>, node: NodeBoxed) -> MergingResult;
    fn launch_settings(&self) -> &LaunchSettings;
}

#[derive(new)]
struct TwoNodesSubGraph {
    graph_1: SubGraphBoxed,
    graph_2: SubGraphBoxed,
    settings: LaunchSettings,
}

#[derive(new)]
struct SingleNodeSubGraph {
    node: NodeBoxed,
    setting: LaunchSettings,
}

#[derive(Default)]
struct EmptySubGraph {
    setting: LaunchSettings,
}

impl Default for LaunchSettings {
    fn default() -> Self {
        Self {
            workgroup_size: WorkgroupSizeSettings::Any,
            workgroup: WorkgroupSettings::Elemwise,
        }
    }
}

impl SubGraph for TwoNodesSubGraph {
    fn register(mut self: Box<Self>, node: NodeBoxed) -> MergingResult {
        let register = |graph: SubGraphBoxed, node: Option<NodeBoxed>| match node {
            Some(node_val) => match graph.register(node_val) {
                MergingResult::Fused(graph) => (graph, None),
                MergingResult::NotFused(graph, node_val) => (graph, Some(node_val)),
            },
            None => (graph, None),
        };

        let mut node = Some(node);

        (self.graph_1, node) = register(self.graph_1, node);
        (self.graph_2, node) = register(self.graph_2, node);

        match node {
            Some(node) => MergingResult::NotFused(self, node),
            None => MergingResult::Fused(self),
        }
    }

    fn launch_settings(&self) -> &LaunchSettings {
        &self.settings
    }
}

impl SubGraph for SingleNodeSubGraph {
    fn register(self: Box<Self>, mut node: NodeBoxed) -> MergingResult {
        for input in self.node.inputs() {
            node.visit_input(input);
        }
        for output in self.node.outputs() {
            node.visit_output(output);
        }

        if node.any_output_visited() {
            return MergingResult::NotFused(self, node);
        }

        if node.all_input_visited() && self.setting.is_compatible_with(&node.launch_settings()) {
            node.reset();

            let settings = self.setting.clone();

            return MergingResult::Fused(Box::new(TwoNodesSubGraph::new(
                self,
                Box::new(SingleNodeSubGraph::new(node, settings.clone())),
                settings,
            )));
        }

        MergingResult::NotFused(self, node)
    }

    fn launch_settings(&self) -> &LaunchSettings {
        &self.setting
    }
}

impl SubGraph for EmptySubGraph {
    fn register(self: Box<Self>, node: NodeBoxed) -> MergingResult {
        let settings = node.launch_settings();
        MergingResult::Fused(Box::new(SingleNodeSubGraph::new(node, settings)))
    }

    fn launch_settings(&self) -> &LaunchSettings {
        &self.setting
    }
}

pub enum MergingResult {
    Fused(SubGraphBoxed),
    NotFused(SubGraphBoxed, NodeBoxed),
}

struct FloatAddOp {
    desc: BinaryOperationDescription,
    settings: LaunchSettings,
    inputs: Vec<TensorId>,
    outputs: Vec<TensorId>,
    inputs_visited: Vec<TensorId>,
    outputs_visited: Vec<TensorId>,
}

impl FloatAddOp {
    pub fn new(desc: BinaryOperationDescription) -> Self {
        let inputs = vec![desc.lhs.id.clone(), desc.rhs.id.clone()];
        let outputs = vec![desc.out.id.clone()];

        Self {
            desc,
            settings: LaunchSettings {
                workgroup_size: WorkgroupSizeSettings::Any,
                workgroup: WorkgroupSettings::Elemwise,
            },
            inputs,
            outputs,
            inputs_visited: Vec::new(),
            outputs_visited: Vec::new(),
        }
    }
}

impl Node for FloatAddOp {
    fn inputs(&self) -> &[TensorId] {
        &self.inputs
    }

    fn outputs(&self) -> &[TensorId] {
        &self.outputs
    }

    fn visit_input(&mut self, input: &TensorId) {
        if self.inputs.contains(input) && !self.inputs_visited.contains(input) {
            self.inputs_visited.push(input.clone());
        }
    }

    fn visit_output(&mut self, output: &TensorId) {
        if self.outputs.contains(output) && !self.outputs_visited.contains(output) {
            self.outputs_visited.push(output.clone());
        }
    }

    fn reset(&mut self) {
        self.inputs_visited.clear();
        self.outputs_visited.clear();
    }

    fn all_input_visited(&self) -> bool {
        self.inputs.len() == self.inputs_visited.len()
    }

    fn any_output_visited(&self) -> bool {
        !self.outputs_visited.is_empty()
    }

    fn launch_settings(&self) -> LaunchSettings {
        self.settings.clone()
    }

    fn register(&self, builder: &mut TraceBuilder) {
        let lhs = builder.input(&self.desc.lhs);
        let rhs = builder.input(&self.desc.rhs);
        let out = builder.output(&self.desc.out);

        struct Proc {
            lhs: Variable,
            rhs: Variable,
            out: Variable,
        }

        impl LazyProcedure for Proc {
            fn expand(&self, scope: &mut Scope, position: Option<Variable>) -> Variable {
                let position = position.unwrap_or(Variable::Id);

                let lhs_input = self.lhs;
                let rhs_input = self.rhs;

                let lhs = scope.create_local(self.lhs.item());
                let rhs = scope.create_local(self.rhs.item());
                let out = self.out;

                gpu!(scope, lhs = lhs_input[position]);
                gpu!(scope, rhs = rhs_input[position]);
                gpu!(scope, out = lhs + rhs);

                out
            }
        }

        builder.register_lazy(lhs.index().unwrap(), Proc { lhs, rhs, out });
    }
}

pub type SubGraphBoxed = Box<dyn SubGraph>;
pub type NodeBoxed = Box<dyn Node>;

pub struct GraphBuilder<R: Runtime> {
    subgraph: SubGraphBoxed,
    status: OptimizationStatus,
    device: R::Device,
    size: usize,
}

impl<R: Runtime> OptimizationBuilder<JitOptimization<R>> for GraphBuilder<R> {
    fn register(&mut self, operation: &OperationDescription) {
        if let OptimizationStatus::Closed = self.status {
            return;
        }

        let node = match operation {
            OperationDescription::NumericFloat(desc) => match desc {
                burn_tensor::repr::NumericOperationDescription::Add(desc) => {
                    Box::new(FloatAddOp::new(desc.clone()))
                }
                _ => {
                    self.status = OptimizationStatus::Closed;
                    return;
                }
            },
            _ => {
                self.status = OptimizationStatus::Closed;
                return;
            }
        };

        let mut subgraph: SubGraphBoxed = Box::new(EmptySubGraph::default());
        core::mem::swap(&mut subgraph, &mut self.subgraph);

        self.subgraph = match subgraph.register(node) {
            MergingResult::Fused(val) => {
                self.size += 1;
                val
            }
            MergingResult::NotFused(val, _) => {
                self.status = OptimizationStatus::Closed;
                val
            }
        };
    }

    fn build(&self) -> JitOptimization<R> {
        todo!()
    }

    fn reset(&mut self) {
        self.size = 0;
        self.subgraph = Box::new(EmptySubGraph::default());
    }

    fn status(&self) -> burn_fusion::OptimizationStatus {
        self.status
    }

    fn properties(&self) -> OptimizationProperties {
        OptimizationProperties {
            ready: true,
            score: self.size as u64,
        }
    }

    fn len(&self) -> usize {
        self.size
    }
}
