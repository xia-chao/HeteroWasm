use waffle::Value;


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dispatch {

    Fixed(u32),

    FromField(usize),

    FromFieldMask(usize, u32),
    FromFieldMaskAdd(usize, u32, u32),
    FromSum(usize, usize),
    FromSumShift(usize, usize, u32),
    FromSubShift(usize, usize, usize, u32),
    FromLoadedShiftMask(usize, usize, u32, u32),
    FromLoaded(usize, u32),
}


#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Artifact<'a> {

    pub wgsl: &'a str,

    pub manifest: String,
}


#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DispatchDecision {

    Gpu,

    CpuFallback { reason: String },
}


#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Kernel {
    pub source: String,
    pub workgroup_size: u32,

    pub dispatch: Dispatch,

    pub fields: Vec<Value>,

    pub min_constant_bytes: i64,

    pub max_constant_bytes: i64,

    pub max_stride_bytes: i64,

    pub index_field: Option<usize>,

    pub launch_count: Option<u32>,

    pub results: Vec<String>,
}


#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DispatchShape {
    pub fields: usize,
    pub results: usize,
}

impl Kernel {

    pub const SHADER_MULS_STILL_SLOWER_AT_OR_BELOW: usize = 31;


    pub fn shader_multiplies_still_slower_than_cpu(&self) -> bool {
        self.source.matches(" * ").count() <= Self::SHADER_MULS_STILL_SLOWER_AT_OR_BELOW
    }


    pub fn required_bytes(&self, params: &[u32]) -> Option<u64> {
        let trip = match self.dispatch {
            Dispatch::Fixed(count) => u64::from(count),
            Dispatch::FromField(field) => u64::from(*params.get(field)?),
            Dispatch::FromFieldMask(field, mask) => u64::from(params.get(field)? & mask),
            Dispatch::FromFieldMaskAdd(field, mask, addend) => {
                u64::from((params.get(field)? & mask).wrapping_add(addend))
            }
            Dispatch::FromSum(left, right) => {
                u64::from(params.get(left)?.wrapping_add(*params.get(right)?))
            }
            Dispatch::FromSumShift(base, shifted, amount) => {
                let shifted = params.get(shifted)?.wrapping_shl(amount);
                u64::from(params.get(base)?.wrapping_add(shifted))
            }
            Dispatch::FromSubShift(base, minuend, subtrahend, amount) => {
                let difference = params.get(minuend)?.wrapping_sub(*params.get(subtrahend)?);
                u64::from(
                    params
                        .get(base)?
                        .wrapping_add(difference.wrapping_shl(amount)),
                )
            }

            Dispatch::FromLoadedShiftMask(_, _, _, mask) => u64::from(mask),

            Dispatch::FromLoaded(_, _) => return None,
        };

        let base = params
            .iter()
            .enumerate()
            .filter(|(slot, _)| {
                !matches!(
                    self.dispatch,
                    Dispatch::FromField(f)
                        | Dispatch::FromFieldMask(f, _)
                        | Dispatch::FromFieldMaskAdd(f, _, _)
                        | Dispatch::FromSum(f, _)
                        | Dispatch::FromSum(_, f)
                        | Dispatch::FromSumShift(f, _, _)
                        | Dispatch::FromSumShift(_, f, _)
                        | Dispatch::FromSubShift(f, _, _, _)
                        | Dispatch::FromSubShift(_, f, _, _)
                        | Dispatch::FromSubShift(_, _, f, _)
                        if *slot == f
                )
            })
            .map(|(_, value)| u64::from(*value))
            .max()
            .unwrap_or(0);
        base.checked_add(trip.checked_mul(4)?)
    }
}

impl Dispatch {

    pub const TRIPS_SLOWER_THAN_CPU_BELOW: u32 = 65_536;


    pub fn fixed_trip_is_slower_than_cpu(self) -> bool {
        match self {
            Dispatch::Fixed(count) => count < Self::TRIPS_SLOWER_THAN_CPU_BELOW,
            _ => false,
        }
    }


    pub fn loaded_count(self, params: &[u32], memory: &[u8]) -> Option<u32> {
        let (bytes, mask) = match self {
            Self::FromLoadedShiftMask(base, shifted, amount, mask) => {
                let base = *params.get(base)?;
                let shifted = *params.get(shifted)?;
                (base.wrapping_add(shifted.wrapping_shl(amount)), mask)
            }
            Self::FromLoaded(slot, offset) => {
                let base = *params.get(slot)?;
                (base.wrapping_add(offset), u32::MAX)
            }
            _ => return None,
        };
        let start = usize::try_from(bytes / 4).ok()?.checked_mul(4)?;
        let end = start.checked_add(4)?;
        let chunk = memory.get(start..end)?;
        let mut buf = [0_u8; 4];
        buf.copy_from_slice(chunk);
        Some(u32::from_le_bytes(buf) & mask)
    }
}


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RewriteOutcome {
    pub loops_removed: usize,
    pub fused: bool,
}


pub struct GpuFusionPlan {
    pub kernel: Kernel,
    pub func: waffle::Func,
}
