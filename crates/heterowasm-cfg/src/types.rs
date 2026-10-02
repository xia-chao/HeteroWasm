use waffle::Block;


#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NaturalLoop {
    pub header: Block,
    pub latches: Vec<Block>,
    pub blocks: Vec<Block>,
}


#[derive(Debug, Clone)]
pub struct ControlFlow {

    pub(crate) reverse_postorder: Vec<Block>,
    pub(crate) immediate_dominator: Vec<Option<Block>>,
    pub(crate) reachable: Vec<bool>,
}
