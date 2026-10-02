use crate::types::{AccessSummary, PairDependence};


pub fn test_pair(first: &AccessSummary, second: &AccessSummary) -> PairDependence {
    if first.base != second.base {
        return PairDependence::NeedsAliasCheck;
    }

    let delta = second.offset.saturating_sub(first.offset);


    if first.stride == second.stride {
        let stride = first.stride;
        if stride == 0 {

            return if delta == 0 {
                PairDependence::Overlap
            } else {
                PairDependence::None
            };
        }
        if delta == 0 {
            return PairDependence::None;
        }
        return if delta.unsigned_abs().is_multiple_of(stride.unsigned_abs()) {
            PairDependence::Overlap
        } else {
            PairDependence::None
        };
    }


    let divisor = gcd(first.stride.unsigned_abs(), second.stride.unsigned_abs());
    if divisor == 0 || !delta.unsigned_abs().is_multiple_of(divisor) {

        PairDependence::None
    } else {
        PairDependence::Overlap
    }
}

fn gcd(left: u64, right: u64) -> u64 {
    if right == 0 {
        left
    } else {
        gcd(right, left % right)
    }
}
