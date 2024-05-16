use std::sync::Arc;

use crate::{
    fusion::{tracing::TraceBuilder, JitOptimization},
    gpu::{gpu, LazyProcedure, Scope, Variable, WorkgroupSize},
    Runtime,
};
use burn_common::id::IdGenerator;
use burn_fusion::{OptimizationBuilder, OptimizationProperties, OptimizationStatus};
use burn_tensor::repr::{BinaryOperationDescription, OperationDescription, TensorId};

use super::{GraphKernelFactory, GraphOptimization};

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
    Elemwise(Vec<usize>),
    Custom,
}

impl LaunchSettings {
    pub fn is_compatible_with(&self, other: &Self) -> bool {
        let workgroup_compatible = self.workgroup == other.workgroup;
        let workgroup_size_compatible = match self.workgroup_size {
            WorkgroupSizeSettings::Any => true,
            WorkgroupSizeSettings::Fixed(workgroup_size) => match other.workgroup_size {
                WorkgroupSizeSettings::Any => true,
                WorkgroupSizeSettings::Fixed(workgroup_size_other) => {
                    workgroup_size == workgroup_size_other
                }
            },
        };

        workgroup_compatible && workgroup_size_compatible
    }

    pub fn most_restrictive(lhs: Self, rhs: Self) -> Self {
        match lhs.workgroup_size {
            WorkgroupSizeSettings::Any => match rhs.workgroup_size {
                WorkgroupSizeSettings::Any => lhs,
                WorkgroupSizeSettings::Fixed(_) => rhs,
            },
            WorkgroupSizeSettings::Fixed(_) => lhs,
        }
    }
}

pub trait Node: Send + Sync {
    fn inputs(&self) -> &[TensorId];
    fn outputs(&self) -> &[TensorId];
    fn visit_input(&mut self, input: &TensorId);
    fn visit_output(&mut self, output: &TensorId);
    fn reset(&mut self);
    fn all_input_visited(&self) -> bool;
    fn any_input_visited(&self) -> bool;
    fn any_output_visited(&self) -> bool;
    fn launch_settings(&self) -> LaunchSettings;
    fn trace(&self, builder: &mut TraceBuilder);
}

pub enum MergeSubGraphResult {
    Merged(SubGraphBoxed),
    Unable(SubGraphBoxed, SubGraphBoxed),
}

pub trait SubGraph: Send + Sync {
    fn register(self: Box<Self>, node: NodeBoxed) -> MergingResult;
    fn merge(self: Box<Self>, other: SubGraphBoxed) -> MergeSubGraphResult;
    fn trace(&self, builder: &mut TraceBuilder);
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

    fn merge(self: Box<Self>, other: SubGraphBoxed) -> MergeSubGraphResult {
        let (graph_2, other) = match self.graph_2.merge(other) {
            MergeSubGraphResult::Merged(graph) => return self.graph_1.merge(graph),
            MergeSubGraphResult::Unable(graph_2, other) => (graph_2, other),
        };

        match self.graph_1.merge(other) {
            MergeSubGraphResult::Merged(graph) => graph.merge(graph_2),
            MergeSubGraphResult::Unable(graph_1, other) => MergeSubGraphResult::Unable(
                Box::new(Self::new(graph_1, graph_2, self.settings)),
                other,
            ),
        }
    }

    fn trace(&self, builder: &mut TraceBuilder) {
        self.graph_1.trace(builder);
        self.graph_2.trace(builder);
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

    fn merge(self: Box<Self>, other: SubGraphBoxed) -> MergeSubGraphResult {
        match other.register(self.node) {
            MergingResult::Fused(graph) => MergeSubGraphResult::Merged(graph),
            MergingResult::NotFused(graph, node) => {
                MergeSubGraphResult::Unable(Box::new(Self::new(node, self.setting)), graph)
            }
        }
    }

    fn trace(&self, builder: &mut TraceBuilder) {
        self.node.trace(builder);
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
        let workgroup = WorkgroupSettings::Elemwise(desc.out.shape.clone());

        Self {
            desc,
            settings: LaunchSettings {
                workgroup_size: WorkgroupSizeSettings::Any,
                workgroup,
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

    fn any_input_visited(&self) -> bool {
        !self.inputs_visited.is_empty()
    }

    fn any_output_visited(&self) -> bool {
        !self.outputs_visited.is_empty()
    }

    fn launch_settings(&self) -> LaunchSettings {
        self.settings.clone()
    }

    fn trace(&self, builder: &mut TraceBuilder) {
        println!("FloatAddOp registering");
        let lhs = builder.input(&self.desc.lhs, Variable::Id);
        let rhs = builder.input(&self.desc.rhs, Variable::Id);
        let out = builder.output(&self.desc.out, Variable::Id);

        struct Proc {
            lhs: Variable,
            rhs: Variable,
            out: Variable,
        }

        impl LazyProcedure for Proc {
            fn expand(&self, scope: &mut Scope, position: Option<Variable>) -> Variable {
                let position = position.unwrap_or(Variable::Id);

                println!("FloatAddOp expand lazy");
                // let lhs = self.lhs;
                // let rhs = self.rhs;
                // let out = self.out;
                let lhs_input = self.lhs;
                let rhs_input = self.rhs;

                let lhs = scope.create_local(self.lhs.item());
                let rhs = scope.create_local(self.rhs.item());
                let out = self.out;

                println!("INPUT {lhs_input:?}");
                // Is local but should not.
                gpu!(scope, lhs = lhs_input[position]);
                gpu!(scope, rhs = rhs_input[position]);
                gpu!(scope, out = lhs + rhs);

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
    graphs: Vec<SubGraphBoxed>,
    launch_settings: Option<LaunchSettings>,
    status: OptimizationStatus,
    device: R::Device,
    size: usize,
}

impl<R: Runtime> GraphBuilder<R> {
    pub fn new(device: R::Device) -> Self {
        Self {
            graphs: Vec::new(),
            launch_settings: None,
            status: OptimizationStatus::Open,
            device,
            size: 0,
        }
    }
    fn add_node(&mut self, node: Box<dyn Node>) {
        if self.graphs.is_empty() {
            let settings = node.launch_settings();
            println!("First node");
            self.graphs = vec![Box::new(SingleNodeSubGraph::new(node, settings))];
            return;
        }

        let mut node_current = Some(node);
        let mut graph_merging_indices = Vec::new();
        let mut merging_done = false;

        self.graphs = self
            .graphs
            .drain(..)
            .into_iter()
            .enumerate()
            .map(|(index, graph)| match node_current.take() {
                Some(node) => {
                    if node.any_input_visited() && !merging_done {
                        // The node is in multiple graphs, we will need to merge them.
                        graph_merging_indices.push(index);
                    }

                    match graph.register(node) {
                        MergingResult::Fused(graph) => {
                            self.size += 1;
                            merging_done = true;
                            graph
                        }
                        MergingResult::NotFused(graph, node) => {
                            self.status = OptimizationStatus::Closed;
                            node_current = Some(node);
                            graph
                        }
                    }
                }
                None => graph,
            })
            .collect();

        if !graph_merging_indices.is_empty() {
            self.merge_subgraphs(graph_merging_indices);
            return;
        }

        if let Some(node) = node_current.take() {
            let node_settings = node.launch_settings();

            if let Some(launch_settings) = &self.launch_settings {
                if launch_settings.is_compatible_with(&node_settings) {
                    let launch_settings =
                        LaunchSettings::most_restrictive(launch_settings.clone(), node_settings);

                    self.graphs.push(Box::new(SingleNodeSubGraph::new(
                        node,
                        launch_settings.clone(),
                    )));

                    self.launch_settings = Some(launch_settings);
                    self.size += 1;
                }
            } else {
                self.graphs.push(Box::new(SingleNodeSubGraph::new(
                    node,
                    node_settings.clone(),
                )));
                self.launch_settings = Some(node_settings);
                self.size += 1;
            }
        } else {
            self.status = OptimizationStatus::Closed;
        }
    }

    fn merge_subgraphs(&mut self, grads_indices: Vec<usize>) {
        let mut graphs = Vec::with_capacity(self.graphs.len() - grads_indices.len() + 1);
        let mut current: Option<SubGraphBoxed> = None;

        for (num, index) in grads_indices.into_iter().enumerate() {
            let graph = self.graphs.remove(index - num);

            current = match current {
                Some(graph_current) => match graph_current.merge(graph) {
                    MergeSubGraphResult::Merged(graph) => Some(graph),
                    MergeSubGraphResult::Unable(_, _) => panic!("Not handled yet"),
                },
                None => Some(graph),
            };
        }
        let graph = current.unwrap();
        graphs.push(graph);

        self.graphs.drain(..).for_each(|graph| graphs.push(graph));
        self.graphs = graphs;
    }
}

impl<R: Runtime> OptimizationBuilder<JitOptimization<R>> for GraphBuilder<R> {
    fn register(&mut self, operation: &OperationDescription) {
        if let OptimizationStatus::Closed = self.status {
            return;
        }

        let node: Box<dyn Node> = match operation {
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

        println!("Registering op {operation:?}");
        self.add_node(node);
    }

    fn build(&self) -> JitOptimization<R> {
        let mut builder = TraceBuilder::new();

        for graph in self.graphs.iter() {
            println!("Trace graph");
            graph.trace(&mut builder);
        }
        println!("Builder");

        let trace = builder.build();
        let info = Arc::new(trace.compiling());

        let grid = match self.launch_settings.as_ref().unwrap().workgroup_size {
            WorkgroupSizeSettings::Any => WorkgroupSize::default(),
            WorkgroupSizeSettings::Fixed(wk) => wk,
        };
        let factory = GraphKernelFactory::new(IdGenerator::generate(), info.clone(), grid);
        let optim = GraphOptimization::new(trace, self.size, self.device.clone(), factory);

        JitOptimization::Graph(optim)
    }

    fn reset(&mut self) {
        self.size = 0;
        self.graphs = vec![];
    }

    fn status(&self) -> burn_fusion::OptimizationStatus {
        self.status
    }

    fn properties(&self) -> OptimizationProperties {
        OptimizationProperties {
            ready: self.size > 0,
            score: self.size as u64,
        }
    }

    fn len(&self) -> usize {
        self.size
    }
}
