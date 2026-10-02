#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispatchSpec {

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

impl DispatchSpec {

    pub fn loaded_count(self, params: &[u32], memory: &[u8]) -> Option<u32> {
        let (start, mask) = self.loaded_word(params)?;
        let end = start.checked_add(4)?;
        let chunk = memory.get(start..end)?;
        let mut buf = [0_u8; 4];
        buf.copy_from_slice(chunk);
        Some(u32::from_le_bytes(buf) & mask)
    }


    pub fn loaded_word(self, params: &[u32]) -> Option<(usize, u32)> {
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
        Some((start, mask))
    }
}


#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shader<'a> {
    pub source: &'a str,
    pub workgroup_size: u32,
    pub dispatch: DispatchSpec,

    pub min_offset_bytes: i64,

    pub max_constant_bytes: i64,
    pub max_stride_bytes: i64,

    pub index_field: Option<usize>,

    pub launch_count: Option<u32>,
}


#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelSpec {
    pub name: String,
    pub source: String,
    pub workgroup_size: u32,
    pub dispatch: DispatchSpec,
    pub uniform_size: usize,

    pub fields: Vec<FieldMapping>,

    pub fields_resolved: bool,

    pub result_slots: usize,

    pub min_offset_bytes: i64,

    pub max_constant_bytes: i64,

    pub max_stride_bytes: i64,

    pub index_field: Option<usize>,

    pub launch_count: Option<u32>,
}


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FieldMapping {
    pub slot: usize,
    pub wasm_param: usize,
}
