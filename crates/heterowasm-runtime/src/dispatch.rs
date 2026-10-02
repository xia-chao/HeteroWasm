use crate::error::RuntimeError;
use crate::types::{DispatchSpec, Shader};


pub(crate) struct Span {
    slot: usize,
    start: u64,
    end: u64,
}


fn trip_count(shader: &Shader<'_>, params: &[u32]) -> Option<u64> {
    match shader.dispatch {
        DispatchSpec::Fixed(count) => Some(u64::from(count)),
        DispatchSpec::FromField(field) => Some(u64::from(*params.get(field)?)),
        DispatchSpec::FromFieldMask(field, mask) => Some(u64::from(params.get(field)? & mask)),
        DispatchSpec::FromFieldMaskAdd(field, mask, addend) => {
            Some(u64::from((params.get(field)? & mask).wrapping_add(addend)))
        }
        DispatchSpec::FromSum(left, right) => Some(u64::from(
            params.get(left)?.wrapping_add(*params.get(right)?),
        )),
        DispatchSpec::FromSumShift(base, shifted, amount) => {
            let shifted = params.get(shifted)?.wrapping_shl(amount);
            Some(u64::from(params.get(base)?.wrapping_add(shifted)))
        }
        DispatchSpec::FromSubShift(base, minuend, subtrahend, amount) => {
            let difference = params.get(minuend)?.wrapping_sub(*params.get(subtrahend)?);
            Some(u64::from(
                params
                    .get(base)?
                    .wrapping_add(difference.wrapping_shl(amount)),
            ))
        }

        DispatchSpec::FromLoadedShiftMask(_, _, _, mask) => Some(u64::from(mask)),

        DispatchSpec::FromLoaded(_, _) => None,
    }
}


pub(crate) fn invocation_count(shader: &Shader<'_>, params: &[u32], dispatched: u32) -> u32 {
    let Some(index_slot) = shader.index_field else {
        return dispatched;
    };
    let (limit_slot, limit_mask) = match shader.dispatch {
        DispatchSpec::FromField(slot) => (slot, u32::MAX),
        DispatchSpec::FromFieldMask(slot, mask) => (slot, mask),
        DispatchSpec::FromFieldMaskAdd(_, _, _)
        | DispatchSpec::FromSum(_, _)
        | DispatchSpec::FromSumShift(_, _, _)
        | DispatchSpec::FromSubShift(_, _, _, _)
        | DispatchSpec::FromLoadedShiftMask(_, _, _, _)
        | DispatchSpec::FromLoaded(_, _)
        | DispatchSpec::Fixed(_) => return dispatched,
    };
    let Ok(stride) = u32::try_from(shader.max_stride_bytes.max(0)) else {
        return dispatched;
    };
    if stride < 4 || stride % 4 != 0 {
        return dispatched;
    }
    let step = stride / 4;
    let Some(&start) = params.get(index_slot) else {
        return dispatched;
    };
    let Some(limit) = params.get(limit_slot).copied() else {
        return dispatched;
    };
    let limit = limit & limit_mask;
    if start <= limit {

        let guard = format!("params.p{index_slot} + 4u * gid.x >= params.p{limit_slot}");
        if shader.source.contains(&guard) {
            let span = limit - start;
            let mut trips = span / stride;
            if span % stride != 0 {
                trips = trips.saturating_add(1);
            }
            return trips;
        }
        return dispatched;
    }

    let span = start - limit;
    let mut trips = span / step;
    if span % step != 0 {
        trips = trips.saturating_add(1);
    }
    dispatched.max(trips)
}


pub(crate) fn scheduled_invocations(shader: &Shader<'_>, params: &[u32], dispatched: u32) -> u32 {
    let from_limit = invocation_count(shader, params, dispatched);
    shader.launch_count.unwrap_or(from_limit).max(from_limit)
}


fn spans(shader: &Shader<'_>, params: &[u32], trip: u64) -> Vec<Span> {

    let stride = u64::try_from(shader.max_stride_bytes.max(0)).unwrap_or(0);
    let constant = u64::try_from(shader.max_constant_bytes.max(0)).unwrap_or(0);

    let upward = Span::upward_bytes(shader, params);
    let bytes = upward.unwrap_or(
        trip.saturating_mul(stride)
            .saturating_add(constant)
            .saturating_add(4),
    );
    params
        .iter()
        .enumerate()
        .filter(|(slot, _)| {
            !matches!(
                shader.dispatch,
                DispatchSpec::FromField(f)
                    | DispatchSpec::FromFieldMask(f, _)
                    | DispatchSpec::FromFieldMaskAdd(f, _, _)
                    | DispatchSpec::FromSum(f, _)
                    | DispatchSpec::FromSum(_, f)
                    | DispatchSpec::FromSumShift(f, _, _)
                    | DispatchSpec::FromSumShift(_, f, _)
                    | DispatchSpec::FromSubShift(f, _, _, _)
                    | DispatchSpec::FromSubShift(_, f, _, _)
                    | DispatchSpec::FromSubShift(_, _, f, _)
                    if *slot == f
            )
        })
        .filter(|(slot, _)| {

            match shader.index_field {
                Some(index) if *slot == index => upward.is_some(),
                _ => true,
            }
        })
        .map(|(slot, value)| Span {
            slot,
            start: u64::from(*value),
            end: u64::from(*value).saturating_add(bytes),
        })
        .collect()
}


pub fn touched_range(shader: &Shader<'_>, params: &[u32], size: u64) -> Option<(u64, u64)> {
    let trip = trip_count(shader, params)?;
    let spans = spans(shader, params, trip);
    if spans.is_empty() {

        let stride = u64::try_from(shader.max_stride_bytes.max(0)).unwrap_or(0);
        let constant = u64::try_from(shader.max_constant_bytes.max(0)).unwrap_or(0);
        let reach = index_reach(shader, params);
        if stride == 0 && constant == 0 && reach == 0 {
            return Some((0, 0));
        }
        let hi = trip
            .saturating_mul(stride)
            .saturating_add(constant)
            .saturating_add(4)
            .saturating_add(reach);
        return Some((0, hi.min(size)));
    }
    let slack = u64::try_from(shader.min_offset_bytes.max(0)).unwrap_or(0);
    let behind = shader.min_offset_bytes.unsigned_abs();
    let lo = spans
        .iter()
        .map(|span| span.start)
        .min()
        .unwrap_or(0)
        .saturating_sub(behind)
        .max(slack);
    let hi = spans
        .iter()
        .map(|span| span.end)
        .max()
        .unwrap_or(0)
        .saturating_add(index_reach(shader, params));
    Some((lo.min(size), hi.min(size)))
}

impl Span {

    fn upward_bytes(shader: &Shader<'_>, params: &[u32]) -> Option<u64> {
        let index_slot = shader.index_field?;
        let (limit_slot, limit_mask) = match shader.dispatch {
            DispatchSpec::FromField(slot) => (slot, u32::MAX),
            DispatchSpec::FromFieldMask(slot, mask) => (slot, mask),
            DispatchSpec::FromFieldMaskAdd(_, _, _)
            | DispatchSpec::FromSum(_, _)
            | DispatchSpec::FromSumShift(_, _, _)
            | DispatchSpec::FromSubShift(_, _, _, _)
            | DispatchSpec::FromLoadedShiftMask(_, _, _, _)
            | DispatchSpec::FromLoaded(_, _)
            | DispatchSpec::Fixed(_) => return None,
        };
        if index_slot == limit_slot {
            return None;
        }
        let stride = u32::try_from(shader.max_stride_bytes.max(0)).ok()?;
        if stride < 4 || stride % 4 != 0 {
            return None;
        }
        let start = *params.get(index_slot)?;
        let limit = params.get(limit_slot).copied()? & limit_mask;
        if start > limit {
            return None;
        }

        let span = limit - start;
        let mut trips = span / stride;
        if span % stride != 0 {
            trips = trips.saturating_add(1);
        }
        let constant = u64::try_from(shader.max_constant_bytes.max(0)).unwrap_or(0);
        Some(
            u64::from(trips)
                .saturating_mul(u64::from(stride))
                .saturating_add(constant)
                .saturating_add(4),
        )
    }


    pub(crate) fn pieces(
        shader: &Shader<'_>,
        params: &[u32],
        size: u64,
    ) -> Option<Vec<(u64, u64)>> {
        let trip = trip_count(shader, params)?;
        let found = spans(shader, params, trip);
        if found.is_empty() {
            let (lo, hi) = touched_range(shader, params, size)?;
            return Some(vec![(lo, hi)]);
        }
        let slack = u64::try_from(shader.min_offset_bytes.max(0)).unwrap_or(0);
        let behind = shader.min_offset_bytes.unsigned_abs();
        let reach = index_reach(shader, params);
        let pieces = found
            .iter()
            .filter_map(|span| {
                let start = span.start.saturating_sub(behind).max(slack).min(size);
                let end = span.end.saturating_add(reach).min(size);
                (end > start).then_some((start, end))
            })
            .collect();
        Some(pieces)
    }
}


fn index_reach(shader: &Shader<'_>, params: &[u32]) -> u64 {

    if Span::upward_bytes(shader, params).is_some() {
        return 0;
    }
    let stride = u64::try_from(shader.max_stride_bytes.max(0)).unwrap_or(0);
    let Some(slot) = shader.index_field else {
        return 0;
    };
    let Some(value) = params.get(slot) else {
        return 0;
    };
    u64::from(*value).saturating_mul(stride)
}


pub fn required_bytes(shader: &Shader<'_>, params: &[u32]) -> Option<u64> {
    let trip = trip_count(shader, params)?;
    spans(shader, params, trip)
        .iter()
        .map(|span| span.end)
        .max()
}


pub fn check(shader: &Shader<'_>, params: &[u32], buffer_bytes: u64) -> Result<(), RuntimeError> {
    let Some(trip) = trip_count(shader, params) else {
        return Err(RuntimeError::DispatchRefused {
            reason: "dispatch count or base unavailable; cannot compute access range".to_string(),
        });
    };

    let spans = spans(shader, params, trip);
    let Some(required) = spans.iter().map(|span| span.end).max() else {
        return Ok(());
    };
    if required > buffer_bytes {
        return Err(RuntimeError::DispatchRefused {
            reason: format!(
                "this access needs {required} bytes, buffer has only {buffer_bytes} — \
                 spec §15 requires falling back to CPU (Wasm would trap; GPU would not). Run the **original unrewritten** wasm."
            ),
        });
    }


    if shader.min_offset_bytes < 0 {
        let slack = shader.min_offset_bytes.unsigned_abs();
        for span in &spans {
            if span.start < slack {
                return Err(RuntimeError::DispatchRefused {
                    reason: format!(
                        "param slot p{} base {} is less than the kernel's required lower-bound headroom {} bytes (const offset {}) — \
                         negative-offset access traps in Wasm but wraps on GPU. Leave enough space before the array, \
                         or run the **original unrewritten** wasm.",
                        span.slot, span.start, slack, shader.min_offset_bytes
                    ),
                });
            }
        }
    }


    for (index, left) in spans.iter().enumerate() {
        for right in spans.iter().skip(index + 1) {
            if left.start == right.start && left.end == right.end {

                continue;
            }
            if left.start < right.end && right.start < left.end {
                return Err(RuntimeError::DispatchRefused {
                    reason: format!(
                        "param slot p{} range [{}, {}) partially overlaps p{} range [{}, {}) — \
                         parallel invocations read the initial memory, while sequential iterations read memory written earlier; \
                         they are not equivalent. Make input/output ranges **exactly equal (in-place) or disjoint**.",
                        left.slot, left.start, left.end, right.slot, right.start, right.end
                    ),
                });
            }
        }
    }

    Ok(())
}
