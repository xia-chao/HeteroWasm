use std::collections::HashSet;

use heterowasm_address::MemoryAccess;
use waffle::Block;


pub fn accesses_in_loop(accesses: &[MemoryAccess], blocks: &[Block]) -> Vec<MemoryAccess> {
    let members: HashSet<Block> = blocks.iter().copied().collect();
    accesses
        .iter()
        .filter(|access| members.contains(&access.block))
        .cloned()
        .collect()
}
