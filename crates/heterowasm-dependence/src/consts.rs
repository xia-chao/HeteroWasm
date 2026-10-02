pub const NOT_SINGLE_TERM: &str =
    "address is not of the form single-base + single-induction + constant";


pub const NON_AFFINE_ADDRESS: &str = "address cannot be reduced to affine form";


pub const OVERLAPPING_ACCESS: &str =
    "same-base access pair fails GCD test; there is intra-loop overlap";


pub const ALIAS_UNPROVEN: &str =
    "access pairs with different bases cannot statically exclude aliasing";


pub const OUTSIDE_COVERAGE: &str =
    "loop is outside Phase 0 committed coverage; addresses not recovered; skip dependence";
