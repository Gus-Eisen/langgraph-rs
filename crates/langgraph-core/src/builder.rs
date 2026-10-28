//! Graph builder and compile-time validation.
//!
//! [`GraphBuilder`] records nodes, edges, and conditional routers.
//! [`GraphBuilder::compile`] checks that structure and returns a
//! [`CompiledGraph`]. Compile does not run nodes or routers.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::future::Future;
use std::pin::Pin;

use crate::error::Error;
use crate::reducer::Reducer;
use crate::state::{same_reducer, GraphState, RESERVED_NAMES};

/// Entry sentinel `__start__`.
///
/// `START` is a legal edge source. It is not a node. A node or a state field
/// cannot use this name. [`GraphBuilder::compile`] requires an edge that
/// leaves `START`.
pub const START: &str = "__start__";

/// Finish sentinel `__end__`.
///
/// `END` is a legal edge target and a legal router destination. It is not a
/// node. No edge may leave `END`.
pub const END: &str = "__end__";

type NodeFuture<'a, S> =
    Pin<Box<dyn Future<Output = Result<<S as GraphState>::Update, Error>> + Send + 'a>>;

/// How the runtime runs a node.
///
/// Sync and async nodes both implement this. The input is `&S` until the Send
/// milestone widens it. Call this trait instead of the stored closure so that
/// widening does not rewrite call sites.
///
/// [`GraphBuilder::compile`] does not call [`InvokeNode::call`].
pub(crate) trait InvokeNode<S>: Send + Sync
where
    S: GraphState,
{
    /// Run the node on the committed snapshot.
    ///
    /// The runtime milestone calls this. Compile only stores the node.
    #[allow(dead_code)]
    fn call<'a>(&'a self, state: &'a S) -> NodeFuture<'a, S>;
}

/// How the runtime runs a conditional router.
///
/// The router sees `reduce(snapshot_at_step_start, this_node_update)` and
/// returns destination names. [`END`] is the finish sentinel. Compile does
/// not call [`InvokeRouter::route`].
pub(crate) trait InvokeRouter<S>: Send + Sync
where
    S: GraphState,
{
    /// Choose the next nodes from the source task's reduced state.
    #[allow(dead_code)]
    fn route(&self, state: &S) -> Result<Vec<String>, Error>;
}

trait AsyncCall<'a, S, F>
where
    S: GraphState + 'a,
    F: ?Sized,
{
    type Fut: Future<Output = Result<S::Update, Error>> + Send + 'a;

    fn call(self, state: &'a S) -> Self::Fut;
}

impl<'a, S, F, Fut> AsyncCall<'a, S, F> for &'a F
where
    S: GraphState + 'a,
    S::Update: Send,
    F: Fn(&'a S) -> Fut + ?Sized,
    Fut: Future<Output = Result<S::Update, Error>> + Send + 'a,
{
    type Fut = Fut;

    fn call(self, state: &'a S) -> Self::Fut {
        self(state)
    }
}

struct SyncNode<F>(F);

impl<S, F> InvokeNode<S> for SyncNode<F>
where
    S: GraphState,
    S::Update: Send + 'static,
    F: Fn(&S) -> Result<S::Update, Error> + Send + Sync,
{
    fn call<'a>(&'a self, state: &'a S) -> NodeFuture<'a, S> {
        Box::pin(std::future::ready((self.0)(state)))
    }
}

struct AsyncNode<F>(F);

impl<S, F> InvokeNode<S> for AsyncNode<F>
where
    S: GraphState + 'static,
    S::Update: Send,
    F: Send + Sync + 'static,
    for<'a> &'a F: AsyncCall<'a, S, F>,
{
    fn call<'a>(&'a self, state: &'a S) -> NodeFuture<'a, S> {
        Box::pin(AsyncCall::call(&self.0, state))
    }
}

/// Marker for a sync node. [`GraphBuilder::add_node`] infers this.
///
/// Name it only when the compiler cannot tell a sync function from an async
/// one: `add_node::<_, SyncNodeMarker>(...)`.
#[derive(Debug, Clone, Copy)]
pub struct SyncNodeMarker;

/// Marker for an async node. [`GraphBuilder::add_node`] infers this.
///
/// Name it only when the compiler cannot tell a sync function from an async
/// one: `add_node::<_, AsyncNodeMarker>(...)`.
#[derive(Debug, Clone, Copy)]
pub struct AsyncNodeMarker;

trait IntoNode<S, Marker>
where
    S: GraphState,
{
    fn into_node(self) -> Box<dyn InvokeNode<S>>;
}

impl<S, F> IntoNode<S, SyncNodeMarker> for F
where
    S: GraphState + 'static,
    S::Update: Send + 'static,
    F: Fn(&S) -> Result<S::Update, Error> + Send + Sync + 'static,
{
    fn into_node(self) -> Box<dyn InvokeNode<S>> {
        Box::new(SyncNode(self))
    }
}

impl<S, F> IntoNode<S, AsyncNodeMarker> for F
where
    S: GraphState + 'static,
    S::Update: Send + 'static,
    F: Send + Sync + 'static,
    for<'a> &'a F: AsyncCall<'a, S, F>,
{
    fn into_node(self) -> Box<dyn InvokeNode<S>> {
        Box::new(AsyncNode(self))
    }
}

struct RouterFn<F>(F);

impl<S, F, T> InvokeRouter<S> for RouterFn<F>
where
    S: GraphState,
    F: Fn(&S) -> Result<T, Error> + Send + Sync,
    T: IntoRoute,
{
    fn route(&self, state: &S) -> Result<Vec<String>, Error> {
        (self.0)(state).map(IntoRoute::into_destinations)
    }
}

mod sealed {
    pub trait EdgeSealed {}
    pub trait RouteSealed {}
}

/// Where an ordinary edge starts.
///
/// [`EdgeSource::One`] is a single node or [`START`]. [`EdgeSource::Wait`] is
/// a waiting edge: the target runs only after every listed node has a
/// completion newer than the one the target last observed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EdgeSource {
    /// One predecessor. [`START`] is the entry sentinel.
    One(String),
    /// Waiting edge. The target is scheduled after every source has completed.
    Wait(Vec<String>),
}

/// Types [`GraphBuilder::add_edge`] accepts as `from`.
///
/// A single name is an ordinary edge. An array, slice, or `Vec` is a waiting
/// edge. This trait is sealed.
pub trait IntoEdgeSource: sealed::EdgeSealed {
    /// Convert this value into one source or a waiting list.
    fn into_edge_source(self) -> EdgeSource;
}

impl sealed::EdgeSealed for EdgeSource {}
impl IntoEdgeSource for EdgeSource {
    fn into_edge_source(self) -> EdgeSource {
        self
    }
}

impl sealed::EdgeSealed for &EdgeSource {}
impl IntoEdgeSource for &EdgeSource {
    fn into_edge_source(self) -> EdgeSource {
        self.clone()
    }
}

impl sealed::EdgeSealed for &str {}
impl IntoEdgeSource for &str {
    fn into_edge_source(self) -> EdgeSource {
        EdgeSource::One(self.to_string())
    }
}

impl sealed::EdgeSealed for String {}
impl IntoEdgeSource for String {
    fn into_edge_source(self) -> EdgeSource {
        EdgeSource::One(self)
    }
}

impl sealed::EdgeSealed for &String {}
impl IntoEdgeSource for &String {
    fn into_edge_source(self) -> EdgeSource {
        EdgeSource::One(self.clone())
    }
}

impl sealed::EdgeSealed for Vec<String> {}
impl IntoEdgeSource for Vec<String> {
    fn into_edge_source(self) -> EdgeSource {
        EdgeSource::Wait(self)
    }
}

impl sealed::EdgeSealed for Vec<&str> {}
impl IntoEdgeSource for Vec<&str> {
    fn into_edge_source(self) -> EdgeSource {
        EdgeSource::Wait(self.into_iter().map(str::to_string).collect())
    }
}

impl<'a> sealed::EdgeSealed for &'a [&'a str] {}
impl<'a> IntoEdgeSource for &'a [&'a str] {
    fn into_edge_source(self) -> EdgeSource {
        EdgeSource::Wait(self.iter().copied().map(str::to_string).collect())
    }
}

impl<const N: usize> sealed::EdgeSealed for [&str; N] {}
impl<const N: usize> IntoEdgeSource for [&str; N] {
    fn into_edge_source(self) -> EdgeSource {
        EdgeSource::Wait(self.into_iter().map(str::to_string).collect())
    }
}

/// Destinations a conditional router returns.
///
/// One name, several names, or [`END`]. A name equal to [`END`] finishes the
/// branch. `Send` is a later milestone and is not represented here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Route {
    destinations: Vec<String>,
}

impl Route {
    /// Route to one node, or to [`END`].
    pub fn to(destination: impl Into<String>) -> Self {
        Self {
            destinations: vec![destination.into()],
        }
    }

    /// Route to [`END`].
    pub fn end() -> Self {
        Self::to(END)
    }

    /// Route to every destination, in this order.
    ///
    /// An empty iterator schedules no nodes. That is not the same as [`END`].
    pub fn to_many<I, T>(destinations: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<String>,
    {
        Self {
            destinations: destinations.into_iter().map(Into::into).collect(),
        }
    }

    /// Node names and [`END`] sentinels, in router order.
    pub fn destinations(&self) -> &[String] {
        &self.destinations
    }
}

/// Return types accepted by [`GraphBuilder::add_conditional_edges`].
///
/// A string is one destination. An array, slice, or `Vec` is several. This
/// trait is sealed.
pub trait IntoRoute: sealed::RouteSealed {
    /// Destination names. [`END`] is the finish sentinel.
    fn into_destinations(self) -> Vec<String>;
}

impl sealed::RouteSealed for Route {}
impl IntoRoute for Route {
    fn into_destinations(self) -> Vec<String> {
        self.destinations
    }
}

impl sealed::RouteSealed for &Route {}
impl IntoRoute for &Route {
    fn into_destinations(self) -> Vec<String> {
        self.destinations.clone()
    }
}

impl sealed::RouteSealed for &str {}
impl IntoRoute for &str {
    fn into_destinations(self) -> Vec<String> {
        vec![self.to_string()]
    }
}

impl sealed::RouteSealed for String {}
impl IntoRoute for String {
    fn into_destinations(self) -> Vec<String> {
        vec![self]
    }
}

impl sealed::RouteSealed for Vec<String> {}
impl IntoRoute for Vec<String> {
    fn into_destinations(self) -> Vec<String> {
        self
    }
}

impl sealed::RouteSealed for Vec<&str> {}
impl IntoRoute for Vec<&str> {
    fn into_destinations(self) -> Vec<String> {
        self.into_iter().map(str::to_string).collect()
    }
}

impl<'a> sealed::RouteSealed for &'a [&'a str] {}
impl<'a> IntoRoute for &'a [&'a str] {
    fn into_destinations(self) -> Vec<String> {
        self.iter().copied().map(str::to_string).collect()
    }
}

impl<const N: usize> sealed::RouteSealed for [&str; N] {}
impl<const N: usize> IntoRoute for [&str; N] {
    fn into_destinations(self) -> Vec<String> {
        self.into_iter().map(str::to_string).collect()
    }
}

/// One ordinary or waiting edge, in registration order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Edge {
    pub(crate) source: EdgeSource,
    pub(crate) target: String,
}

struct RecordedConditional<S: GraphState> {
    from: String,
    router: Box<dyn InvokeRouter<S>>,
    path_map: Option<Vec<(String, String)>>,
}

/// A conditional router stored on a [`CompiledGraph`].
///
/// `path_map` is `None` when the router returns node names directly. Unknown
/// names then fail at runtime, not in [`GraphBuilder::compile`]. When the map
/// is present, its values were checked at compile. Keys are router labels,
/// not node names.
pub(crate) struct ConditionalEdge<S: GraphState> {
    pub(crate) from: String,
    pub(crate) router: Box<dyn InvokeRouter<S>>,
    pub(crate) path_map: Option<BTreeMap<String, String>>,
}

impl<S: GraphState> fmt::Debug for ConditionalEdge<S> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let _router = &self.router;
        formatter
            .debug_struct("ConditionalEdge")
            .field("from", &self.from)
            .field("path_map", &self.path_map)
            .finish_non_exhaustive()
    }
}

/// Recorded graph structure for `S`.
///
/// [`GraphBuilder::compile`] produces this. It does not run the graph, and it
/// has no checkpointer and no interrupt breakpoints.
pub struct CompiledGraph<S: GraphState> {
    pub(crate) nodes: Vec<(String, Box<dyn InvokeNode<S>>)>,
    pub(crate) edges: Vec<Edge>,
    pub(crate) conditional_edges: Vec<ConditionalEdge<S>>,
}

impl<S: GraphState> fmt::Debug for CompiledGraph<S> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CompiledGraph")
            .field("nodes", &node_names(&self.nodes))
            .field("edges", &self.edges)
            .field("conditional_edges", &self.conditional_edges)
            .finish()
    }
}

/// Builds a graph for `S` and checks it at [`GraphBuilder::compile`].
///
/// Nodes, edges, and routers are stored. Nothing here runs them.
pub struct GraphBuilder<S: GraphState> {
    nodes: Vec<(String, Box<dyn InvokeNode<S>>)>,
    edges: Vec<Edge>,
    conditional: Vec<RecordedConditional<S>>,
}

impl<S: GraphState> Default for GraphBuilder<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: GraphState> GraphBuilder<S> {
    /// Empty builder. `S::schema` is read at [`Self::compile`].
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            conditional: Vec::new(),
        }
    }

    /// Record a node.
    ///
    /// `node` is a sync function `fn(&S) -> Result<S::Update, Error>` or an
    /// async function that returns
    /// `Future<Output = Result<S::Update, Error>> + Send`. Both are stored
    /// type-erased. This does not call `node`.
    ///
    /// Names are unique. [`Self::compile`] rejects a duplicate, an empty
    /// name, or a reserved name. `S` and `S::Update` are `'static`, and
    /// `S::Update` is `Send`, so the stored node can later run on tokio.
    #[allow(private_bounds)]
    pub fn add_node<F, M>(&mut self, name: impl Into<String>, node: F) -> &mut Self
    where
        S: 'static,
        S::Update: Send + 'static,
        F: IntoNode<S, M>,
    {
        self.nodes.push((name.into(), node.into_node()));
        self
    }

    /// Record an ordinary edge, or a waiting edge when `from` is a list.
    ///
    /// [`START`] and [`END`] are legal endpoints. [`Self::compile`] rejects
    /// an edge that leaves [`END`], enters [`START`], or names an unknown
    /// node. A list is a waiting edge even when it has one entry. This does
    /// not traverse the edge.
    pub fn add_edge(&mut self, from: impl IntoEdgeSource, to: impl Into<String>) -> &mut Self {
        self.edges.push(Edge {
            source: from.into_edge_source(),
            target: to.into(),
        });
        self
    }

    /// Record a conditional router.
    ///
    /// The router is stored and not called. It will later see the source
    /// node's own reduced state and return one node name, several node names,
    /// or [`END`]. Without a path map, an unknown name fails at runtime.
    /// [`Self::add_conditional_edges_with_path_map`] checks static targets at
    /// compile.
    pub fn add_conditional_edges<R, T>(&mut self, from: impl Into<String>, router: R) -> &mut Self
    where
        R: Fn(&S) -> Result<T, Error> + Send + Sync + 'static,
        T: IntoRoute,
    {
        self.push_conditional(from.into(), router, None)
    }

    /// Record a conditional router and a static path map.
    ///
    /// Map keys are the labels the router returns. Map values are node names
    /// or [`END`]. [`Self::compile`] rejects a value that is neither. Keys
    /// are not node names and are not checked. Duplicate keys are rejected.
    /// The router is not called.
    pub fn add_conditional_edges_with_path_map<R, T, I, K, V>(
        &mut self,
        from: impl Into<String>,
        router: R,
        path_map: I,
    ) -> &mut Self
    where
        R: Fn(&S) -> Result<T, Error> + Send + Sync + 'static,
        T: IntoRoute,
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        let pairs = path_map
            .into_iter()
            .map(|(key, target)| (key.into(), target.into()))
            .collect();
        self.push_conditional(from.into(), router, Some(pairs))
    }

    fn push_conditional<R, T>(
        &mut self,
        from: String,
        router: R,
        path_map: Option<Vec<(String, String)>>,
    ) -> &mut Self
    where
        R: Fn(&S) -> Result<T, Error> + Send + Sync + 'static,
        T: IntoRoute,
    {
        self.conditional.push(RecordedConditional {
            from,
            router: Box::new(RouterFn(router)),
            path_map,
        });
        self
    }

    /// Check the recorded graph and return it.
    ///
    /// This does not run nodes or routers. There is no checkpointer parameter
    /// and no interrupt breakpoint parameter.
    ///
    /// Compile rejects:
    ///
    /// - no outgoing edge from [`START`]
    /// - an edge whose source is not [`START`] and not a known node
    /// - an edge whose target is not [`END`] and not a known node
    /// - any edge that leaves [`END`] or enters [`START`]
    /// - a static path-map target that is neither [`END`] nor a known node
    /// - a duplicate node name, or a node or field that uses a reserved name
    /// - an empty node or field name
    /// - two registrations of the same field, including two reducer kinds
    /// - a waiting edge with no sources
    /// - a duplicate path-map key
    ///
    /// Cycles are ordinary edges. Unreachable nodes are allowed. Two `Custom`
    /// reducers on one field are the same kind: function pointers are not
    /// compared, and the duplicate name is still rejected.
    pub fn compile(self) -> Result<CompiledGraph<S>, Error> {
        validate_schema::<S>()?;
        let node_names = validate_nodes(&self.nodes)?;
        validate_edges(&self.edges, &node_names)?;
        let conditional_edges = validate_conditional(self.conditional, &node_names)?;
        if !leaves_start(&self.edges, &conditional_edges) {
            return Err(Error::GraphValidation("START has no outgoing edge".into()));
        }
        Ok(CompiledGraph {
            nodes: self.nodes,
            edges: self.edges,
            conditional_edges,
        })
    }
}

impl<S: GraphState> fmt::Debug for GraphBuilder<S> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let conditional: Vec<String> = self
            .conditional
            .iter()
            .map(|edge| {
                let _router = &edge.router;
                match &edge.path_map {
                    None => format!("{} (no path map)", edge.from),
                    Some(pairs) => format!("{} {pairs:?}", edge.from),
                }
            })
            .collect();
        formatter
            .debug_struct("GraphBuilder")
            .field("nodes", &node_names(&self.nodes))
            .field("edges", &self.edges)
            .field("conditional_edges", &conditional)
            .finish()
    }
}

fn node_names<S: GraphState>(nodes: &[(String, Box<dyn InvokeNode<S>>)]) -> Vec<&str> {
    nodes
        .iter()
        .map(|(name, node)| {
            let _stored = node;
            name.as_str()
        })
        .collect()
}

fn validate_schema<S: GraphState>() -> Result<(), Error> {
    let mut seen: BTreeMap<&str, Reducer> = BTreeMap::new();
    for field in S::schema() {
        if field.name.is_empty() {
            return Err(Error::GraphValidation("channel name is empty".into()));
        }
        if RESERVED_NAMES.contains(&field.name) {
            return Err(Error::GraphValidation(format!(
                "channel name '{}' is reserved",
                field.name
            )));
        }
        if let Some(previous) = seen.get(field.name) {
            if same_reducer(*previous, field.reducer) {
                return Err(Error::GraphValidation(format!(
                    "duplicate channel '{}'",
                    field.name
                )));
            }
            return Err(Error::GraphValidation(format!(
                "channel '{}' is registered with two reducers",
                field.name
            )));
        }
        seen.insert(field.name, field.reducer);
    }
    Ok(())
}

fn validate_nodes<S: GraphState>(
    nodes: &[(String, Box<dyn InvokeNode<S>>)],
) -> Result<BTreeSet<&str>, Error> {
    let mut names = BTreeSet::new();
    for (name, node) in nodes {
        let _stored = node;
        if name.is_empty() {
            return Err(Error::GraphValidation("node name is empty".into()));
        }
        if RESERVED_NAMES.contains(&name.as_str()) {
            return Err(Error::GraphValidation(format!(
                "node name '{name}' is reserved"
            )));
        }
        if !names.insert(name.as_str()) {
            return Err(Error::GraphValidation(format!("duplicate node '{name}'")));
        }
    }
    Ok(names)
}

fn validate_edges(edges: &[Edge], nodes: &BTreeSet<&str>) -> Result<(), Error> {
    for edge in edges {
        validate_edge(edge, nodes)?;
    }
    Ok(())
}

fn validate_edge(edge: &Edge, nodes: &BTreeSet<&str>) -> Result<(), Error> {
    match &edge.source {
        EdgeSource::One(name) => check_source(name, nodes)?,
        EdgeSource::Wait(names) => {
            if names.is_empty() {
                return Err(Error::GraphValidation("waiting edge has no sources".into()));
            }
            for name in names {
                check_source(name, nodes)?;
            }
        }
    }
    check_target(&edge.target, nodes)
}

fn validate_conditional<S: GraphState>(
    conditional: Vec<RecordedConditional<S>>,
    nodes: &BTreeSet<&str>,
) -> Result<Vec<ConditionalEdge<S>>, Error> {
    let mut compiled = Vec::with_capacity(conditional.len());
    for edge in conditional {
        check_conditional_source(&edge.from, nodes)?;
        let path_map = match edge.path_map {
            None => None,
            Some(pairs) => Some(checked_path_map(pairs, nodes)?),
        };
        compiled.push(ConditionalEdge {
            from: edge.from,
            router: edge.router,
            path_map,
        });
    }
    Ok(compiled)
}

fn checked_path_map(
    pairs: Vec<(String, String)>,
    nodes: &BTreeSet<&str>,
) -> Result<BTreeMap<String, String>, Error> {
    let mut map = BTreeMap::new();
    for (key, target) in pairs {
        if target == START {
            return Err(Error::GraphValidation(format!(
                "path map target '{target}' enters START"
            )));
        }
        if target != END && !nodes.contains(target.as_str()) {
            return Err(Error::GraphValidation(format!(
                "path map target '{target}' is not a node"
            )));
        }
        if map.contains_key(&key) {
            return Err(Error::GraphValidation(format!(
                "duplicate path map key '{key}'"
            )));
        }
        map.insert(key, target);
    }
    Ok(map)
}

fn check_source(source: &str, nodes: &BTreeSet<&str>) -> Result<(), Error> {
    if source == END {
        return Err(Error::GraphValidation("edge leaves END".into()));
    }
    if source == START || nodes.contains(source) {
        return Ok(());
    }
    Err(Error::GraphValidation(format!(
        "edge source '{source}' is not a node"
    )))
}

fn check_target(target: &str, nodes: &BTreeSet<&str>) -> Result<(), Error> {
    if target == START {
        return Err(Error::GraphValidation("edge enters START".into()));
    }
    if target == END || nodes.contains(target) {
        return Ok(());
    }
    Err(Error::GraphValidation(format!(
        "edge target '{target}' is not a node"
    )))
}

fn check_conditional_source(source: &str, nodes: &BTreeSet<&str>) -> Result<(), Error> {
    if source == END {
        return Err(Error::GraphValidation("conditional edge leaves END".into()));
    }
    if source == START || nodes.contains(source) {
        return Ok(());
    }
    Err(Error::GraphValidation(format!(
        "conditional edge source '{source}' is not a node"
    )))
}

fn leaves_start<S: GraphState>(edges: &[Edge], conditional: &[ConditionalEdge<S>]) -> bool {
    edges.iter().any(|edge| source_leaves_start(&edge.source))
        || conditional.iter().any(|edge| edge.from == START)
}

fn source_leaves_start(source: &EdgeSource) -> bool {
    match source {
        EdgeSource::One(name) => name == START,
        EdgeSource::Wait(names) => names.iter().any(|name| name == START),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::future::Future;

    use serde_json::Value;

    use super::*;
    use crate::{Field, Reducer};

    struct Count;

    impl GraphState for Count {
        type Update = ();

        fn schema() -> &'static [Field] {
            static SCHEMA: &[Field] = &[Field {
                name: "n",
                reducer: Reducer::Overwrite,
            }];
            SCHEMA
        }

        fn to_channel_values(&self) -> Result<BTreeMap<String, Value>, Error> {
            Ok(BTreeMap::from([("n".to_string(), Value::from(0))]))
        }

        fn update_to_channel_values(
            _update: &Self::Update,
        ) -> Result<BTreeMap<String, Value>, Error> {
            Ok(BTreeMap::new())
        }

        fn from_channel_values(_values: &BTreeMap<String, Value>) -> Result<Self, Error> {
            Ok(Count)
        }
    }

    macro_rules! schema_state {
        ($ty:ident, $($field:expr),+ $(,)?) => {
            struct $ty;

            impl GraphState for $ty {
                type Update = ();

                fn schema() -> &'static [Field] {
                    static SCHEMA: &[Field] = &[$($field),+];
                    SCHEMA
                }

                fn to_channel_values(&self) -> Result<BTreeMap<String, Value>, Error> {
                    let _ = self;
                    Ok(BTreeMap::new())
                }

                fn update_to_channel_values(
                    _update: &Self::Update,
                ) -> Result<BTreeMap<String, Value>, Error> {
                    Ok(BTreeMap::new())
                }

                fn from_channel_values(
                    _values: &BTreeMap<String, Value>,
                ) -> Result<Self, Error> {
                    Ok($ty)
                }
            }
        };
    }

    schema_state!(
        ReservedField,
        Field {
            name: "__error__",
            reducer: Reducer::Overwrite,
        },
    );

    schema_state!(
        TwoReducers,
        Field {
            name: "n",
            reducer: Reducer::Overwrite,
        },
        Field {
            name: "n",
            reducer: Reducer::Append,
        },
    );

    schema_state!(
        SameReducer,
        Field {
            name: "n",
            reducer: Reducer::Overwrite,
        },
        Field {
            name: "n",
            reducer: Reducer::Overwrite,
        },
    );

    fn custom_left(_current: Option<&Value>, update: &Value) -> Result<Value, Error> {
        Ok(update.clone())
    }

    fn custom_right(_current: Option<&Value>, update: &Value) -> Result<Value, Error> {
        Ok(update.clone())
    }

    schema_state!(
        TwoCustoms,
        Field {
            name: "n",
            reducer: Reducer::Custom(custom_left),
        },
        Field {
            name: "n",
            reducer: Reducer::Custom(custom_right),
        },
    );

    schema_state!(
        EmptyField,
        Field {
            name: "",
            reducer: Reducer::Overwrite,
        },
    );

    fn panic_node(_state: &Count) -> Result<(), Error> {
        panic!("compile must not run nodes");
    }

    async fn panic_async(_state: &Count) -> Result<(), Error> {
        panic!("compile must not run async nodes");
    }

    fn deferred(state: &Count) -> impl Future<Output = Result<(), Error>> + Send + '_ {
        let _ = state;
        async { panic!("compile must not run deferred nodes") }
    }

    fn panic_router(_state: &Count) -> Result<&'static str, Error> {
        panic!("compile must not run routers");
    }

    fn validation_message<S: GraphState>(result: Result<CompiledGraph<S>, Error>) -> String {
        match result.expect_err("compile should reject the graph") {
            Error::GraphValidation(message) => message,
            other => panic!("expected GraphValidation, got {other:?}"),
        }
    }

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn sentinels_match_reserved_names() {
        assert_eq!(START, "__start__");
        assert_eq!(END, "__end__");
        assert!(RESERVED_NAMES.contains(&START));
        assert!(RESERVED_NAMES.contains(&END));
        assert_eq!(Route::end().destinations(), [END.to_string()]);
        assert_eq!(Route::to("a").destinations(), ["a".to_string()]);
        assert_eq!(
            Route::to_many(["a", END]).destinations(),
            ["a".to_string(), END.to_string()]
        );
    }

    #[test]
    fn compiled_graph_is_send_and_sync() {
        assert_send_sync::<GraphBuilder<Count>>();
        assert_send_sync::<CompiledGraph<Count>>();
    }

    #[test]
    fn compile_accepts_a_cycle() {
        let mut builder = GraphBuilder::<Count>::new();
        builder
            .add_node("a", panic_node)
            .add_node("b", panic_async)
            .add_edge(START, "a")
            .add_edge("a", "b")
            .add_edge("b", "a");

        let compiled = builder.compile().expect("cycle");
        assert_eq!(
            compiled
                .nodes
                .iter()
                .map(|(name, _node)| name.as_str())
                .collect::<Vec<_>>(),
            ["a", "b"]
        );
        assert_eq!(
            compiled.edges,
            [
                Edge {
                    source: EdgeSource::One(START.to_string()),
                    target: "a".to_string(),
                },
                Edge {
                    source: EdgeSource::One("a".to_string()),
                    target: "b".to_string(),
                },
                Edge {
                    source: EdgeSource::One("b".to_string()),
                    target: "a".to_string(),
                },
            ]
        );
        let rendered = format!("{compiled:?}");
        assert!(rendered.contains("a"), "{rendered}");
        assert!(rendered.contains("b"), "{rendered}");
    }

    #[test]
    fn compile_stores_every_node_shape_without_running_it() {
        let mut builder = GraphBuilder::<Count>::new();
        let captured = 1;
        builder
            .add_node("sync", panic_node)
            .add_node("async_fn", panic_async)
            .add_node("deferred", deferred)
            .add_node("closure", |_state: &Count| -> Result<(), Error> {
                panic!("compile must not run closures");
            })
            .add_node("captured", move |_state: &Count| -> Result<(), Error> {
                let _ = captured;
                panic!("compile must not run captured closures");
            })
            .add_edge(START, "sync")
            .add_edge("sync", END);

        let compiled = builder.compile().expect("stored nodes");
        let rendered = format!("{compiled:?}");
        for name in ["sync", "async_fn", "deferred", "closure", "captured"] {
            assert!(rendered.contains(name), "{rendered}");
        }
    }

    #[test]
    fn compile_allows_an_unreachable_node() {
        let mut builder = GraphBuilder::<Count>::new();
        builder
            .add_node("a", panic_node)
            .add_node("lonely", panic_node)
            .add_edge(START, "a")
            .add_edge("a", END);

        let compiled = builder.compile().expect("unreachable node");
        assert_eq!(compiled.nodes.len(), 2);
    }

    #[test]
    fn compile_rejects_a_missing_entry_edge() {
        let mut builder = GraphBuilder::<Count>::new();
        builder.add_node("a", panic_node).add_edge("a", END);

        assert_eq!(
            validation_message(builder.compile()),
            "START has no outgoing edge"
        );
    }

    #[test]
    fn compile_rejects_an_edge_out_of_end() {
        let mut builder = GraphBuilder::<Count>::new();
        builder
            .add_node("a", panic_node)
            .add_edge(START, "a")
            .add_edge("a", END)
            .add_edge(END, "a");

        assert_eq!(validation_message(builder.compile()), "edge leaves END");
    }

    #[test]
    fn compile_rejects_an_edge_into_start() {
        let mut builder = GraphBuilder::<Count>::new();
        builder
            .add_node("a", panic_node)
            .add_edge(START, "a")
            .add_edge("a", START);

        assert_eq!(validation_message(builder.compile()), "edge enters START");
    }

    #[test]
    fn compile_rejects_an_unknown_target() {
        let mut builder = GraphBuilder::<Count>::new();
        builder.add_edge(START, "missing");

        assert_eq!(
            validation_message(builder.compile()),
            "edge target 'missing' is not a node"
        );
    }

    #[test]
    fn compile_rejects_an_unknown_source() {
        let mut builder = GraphBuilder::<Count>::new();
        builder
            .add_node("a", panic_node)
            .add_edge(START, "a")
            .add_edge("ghost", "a");

        assert_eq!(
            validation_message(builder.compile()),
            "edge source 'ghost' is not a node"
        );
    }

    #[test]
    fn compile_rejects_a_duplicate_node_name() {
        let mut builder = GraphBuilder::<Count>::new();
        builder
            .add_node("a", panic_node)
            .add_node("a", panic_async)
            .add_edge(START, "a")
            .add_edge("a", END);

        assert_eq!(validation_message(builder.compile()), "duplicate node 'a'");
    }

    #[test]
    fn compile_rejects_a_reserved_node_name() {
        let mut builder = GraphBuilder::<Count>::new();
        builder
            .add_node("__interrupt__", panic_node)
            .add_edge(START, "__interrupt__")
            .add_edge("__interrupt__", END);

        assert_eq!(
            validation_message(builder.compile()),
            "node name '__interrupt__' is reserved"
        );
    }

    #[test]
    fn compile_rejects_an_empty_node_name() {
        let mut builder = GraphBuilder::<Count>::new();
        builder.add_node("", panic_node).add_edge(START, END);

        assert_eq!(validation_message(builder.compile()), "node name is empty");
    }

    #[test]
    fn compile_rejects_a_reserved_field_name() {
        let mut builder = GraphBuilder::<ReservedField>::new();
        builder.add_edge(START, END);

        assert_eq!(
            validation_message(builder.compile()),
            "channel name '__error__' is reserved"
        );
    }

    #[test]
    fn compile_rejects_two_reducer_kinds_on_one_field() {
        let mut builder = GraphBuilder::<TwoReducers>::new();
        builder.add_edge(START, END);

        assert_eq!(
            validation_message(builder.compile()),
            "channel 'n' is registered with two reducers"
        );
    }

    #[test]
    fn compile_rejects_a_duplicate_field_of_the_same_reducer_kind() {
        let mut builder = GraphBuilder::<SameReducer>::new();
        builder.add_edge(START, END);

        assert_eq!(
            validation_message(builder.compile()),
            "duplicate channel 'n'"
        );
    }

    #[test]
    fn compile_treats_custom_reducers_as_one_kind() {
        let mut builder = GraphBuilder::<TwoCustoms>::new();
        builder.add_edge(START, END);

        assert_eq!(
            validation_message(builder.compile()),
            "duplicate channel 'n'"
        );
    }

    #[test]
    fn compile_rejects_an_empty_field_name() {
        let mut builder = GraphBuilder::<EmptyField>::new();
        builder.add_edge(START, END);

        assert_eq!(
            validation_message(builder.compile()),
            "channel name is empty"
        );
    }

    #[test]
    fn compile_accepts_a_waiting_edge_and_start_to_end() {
        let mut builder = GraphBuilder::<Count>::new();
        builder
            .add_node("a", panic_node)
            .add_node("b", panic_node)
            .add_node("join", panic_async)
            .add_edge(START, "a")
            .add_edge(START, "b")
            .add_edge(["a", "b"], "join")
            .add_edge(Vec::from(["join".to_string()]), END)
            .add_edge(EdgeSource::One("join".to_string()), END);

        let compiled = builder.compile().expect("waiting edge");
        assert_eq!(
            compiled.edges[2].source,
            EdgeSource::Wait(vec!["a".to_string(), "b".to_string()])
        );
        assert_eq!(
            compiled.edges[3].source,
            EdgeSource::Wait(vec!["join".to_string()])
        );

        let mut direct = GraphBuilder::<Count>::new();
        direct.add_edge(START, END);
        direct.compile().expect("START to END");
    }

    #[test]
    fn start_inside_a_waiting_edge_counts_as_an_entry() {
        let mut builder = GraphBuilder::<Count>::new();
        builder
            .add_node("a", panic_node)
            .add_node("b", panic_node)
            .add_edge([START, "a"], "b")
            .add_edge("b", END);

        builder.compile().expect("waiting entry");
    }

    #[test]
    fn compile_rejects_a_waiting_edge_with_no_sources() {
        let mut builder = GraphBuilder::<Count>::new();
        builder
            .add_node("a", panic_node)
            .add_edge(START, "a")
            .add_edge(Vec::<&str>::new(), "a");

        assert_eq!(
            validation_message(builder.compile()),
            "waiting edge has no sources"
        );
    }

    #[test]
    fn compile_rejects_a_waiting_edge_that_leaves_end() {
        let mut builder = GraphBuilder::<Count>::new();
        builder
            .add_node("a", panic_node)
            .add_edge(START, "a")
            .add_edge([END, "a"], "a");

        assert_eq!(validation_message(builder.compile()), "edge leaves END");
    }

    #[test]
    fn compile_stores_a_router_without_running_it() {
        let mut builder = GraphBuilder::<Count>::new();
        builder
            .add_node("a", panic_node)
            .add_node("b", panic_node)
            .add_edge(START, "a")
            .add_edge("b", END)
            .add_conditional_edges("a", panic_router)
            .add_conditional_edges("a", |state: &Count| {
                panic_router(state)?;
                Ok(vec!["b", END])
            })
            .add_conditional_edges("a", |_state: &Count| Ok(Route::end()))
            .add_conditional_edges("a", |_state: &Count| Ok(["b", END]));

        let compiled = builder.compile().expect("routers stored");
        assert_eq!(compiled.conditional_edges.len(), 4);
        assert!(compiled
            .conditional_edges
            .iter()
            .all(|edge| edge.path_map.is_none()));
    }

    #[test]
    fn compile_checks_a_static_path_map_and_not_the_router() {
        let mut builder = GraphBuilder::<Count>::new();
        builder
            .add_node("a", panic_node)
            .add_node("b", panic_node)
            .add_edge(START, "a")
            .add_edge("b", END)
            .add_conditional_edges_with_path_map("a", panic_router, [("go", "b"), ("stop", END)]);

        let compiled = builder.compile().expect("path map");
        let map = compiled.conditional_edges[0]
            .path_map
            .as_ref()
            .expect("map stored");
        assert_eq!(map.get("go").map(String::as_str), Some("b"));
        assert_eq!(map.get("stop").map(String::as_str), Some(END));

        let mut unknown = GraphBuilder::<Count>::new();
        unknown
            .add_node("a", panic_node)
            .add_edge(START, "a")
            .add_edge("a", END)
            .add_conditional_edges_with_path_map("a", panic_router, [("go", "missing")]);
        assert_eq!(
            validation_message(unknown.compile()),
            "path map target 'missing' is not a node"
        );

        let mut into_start = GraphBuilder::<Count>::new();
        into_start
            .add_node("a", panic_node)
            .add_edge(START, "a")
            .add_edge("a", END)
            .add_conditional_edges_with_path_map("a", panic_router, [("go", START)]);
        assert_eq!(
            validation_message(into_start.compile()),
            "path map target '__start__' enters START"
        );

        let mut duplicate_key = GraphBuilder::<Count>::new();
        duplicate_key
            .add_node("a", panic_node)
            .add_edge(START, "a")
            .add_edge("a", END)
            .add_conditional_edges_with_path_map("a", panic_router, [("go", "a"), ("go", END)]);
        assert_eq!(
            validation_message(duplicate_key.compile()),
            "duplicate path map key 'go'"
        );
    }

    #[test]
    fn conditional_from_start_counts_as_an_entry() {
        let mut builder = GraphBuilder::<Count>::new();
        builder
            .add_node("a", panic_node)
            .add_conditional_edges(START, panic_router)
            .add_edge("a", END);

        builder.compile().expect("conditional entry");
    }

    #[test]
    fn compile_rejects_a_conditional_edge_that_leaves_end() {
        let mut builder = GraphBuilder::<Count>::new();
        builder
            .add_node("a", panic_node)
            .add_edge(START, "a")
            .add_edge("a", END)
            .add_conditional_edges(END, panic_router);

        assert_eq!(
            validation_message(builder.compile()),
            "conditional edge leaves END"
        );
    }

    #[test]
    fn compile_rejects_an_unknown_conditional_source() {
        let mut builder = GraphBuilder::<Count>::new();
        builder
            .add_edge(START, END)
            .add_conditional_edges("ghost", panic_router);

        assert_eq!(
            validation_message(builder.compile()),
            "conditional edge source 'ghost' is not a node"
        );
    }

    #[test]
    fn compile_keeps_static_and_conditional_edges_from_one_node() {
        let mut builder = GraphBuilder::<Count>::new();
        builder
            .add_node("a", panic_node)
            .add_node("b", panic_node)
            .add_edge(START, "a")
            .add_edge("a", "b")
            .add_edge("b", END)
            .add_conditional_edges("a", panic_router);

        let compiled = builder.compile().expect("both edge kinds");
        assert_eq!(compiled.edges.len(), 3);
        assert_eq!(compiled.conditional_edges.len(), 1);
    }
}
