use waffle::FunctionBody;

use crate::conformance;
use crate::error::LowerError;
use crate::types::{Artifact, Dispatch, Kernel};


pub fn artifact<'a>(kernel: &'a Kernel, body: &FunctionBody) -> Result<Artifact<'a>, LowerError> {
    let mapping = conformance::field_mapping(body, &kernel.fields);
    let fields = match &mapping {
        Some(indices) => indices
            .iter()
            .enumerate()
            .map(|(slot, parameter)| {
                format!(
                    "    {{ \"slot\": {slot}, \"wasm_param\": {parameter}, \"name\": \"params.p{slot}\" }}"
                )
            })
            .collect::<Vec<_>>()
            .join(",\n"),
        None => String::new(),
    };
    let dispatch = match kernel.dispatch {
        Dispatch::Fixed(count) => format!("{{ \"kind\": \"fixed\", \"count\": {count} }}"),
        Dispatch::FromField(field) => {
            format!("{{ \"kind\": \"from_field\", \"field\": {field} }}")
        }
        Dispatch::FromFieldMask(field, mask) => {
            format!("{{ \"kind\": \"from_field_mask\", \"field\": {field}, \"mask\": {mask} }}")
        }
        Dispatch::FromFieldMaskAdd(field, mask, addend) => format!(
            "{{ \"kind\": \"from_field_mask_add\", \"field\": {field}, \"mask\": {mask}, \"addend\": {addend} }}"
        ),
        Dispatch::FromSum(left, right) => {
            format!("{{ \"kind\": \"from_sum\", \"left\": {left}, \"right\": {right} }}")
        }
        Dispatch::FromSumShift(base, shifted, amount) => format!(
            "{{ \"kind\": \"from_sum_shift\", \"base\": {base}, \"shifted\": {shifted}, \"amount\": {amount} }}"
        ),
        Dispatch::FromSubShift(base, minuend, subtrahend, amount) => format!(
            "{{ \"kind\": \"from_sub_shift\", \"base\": {base}, \"minuend\": {minuend}, \"subtrahend\": {subtrahend}, \"amount\": {amount} }}"
        ),
        Dispatch::FromLoadedShiftMask(base, shifted, amount, mask) => format!(
            "{{ \"kind\": \"from_loaded_shift_mask\", \"base\": {base}, \"shifted\": {shifted}, \"amount\": {amount}, \"mask\": {mask} }}"
        ),
        Dispatch::FromLoaded(slot, offset) => format!(
            "{{ \"kind\": \"from_loaded\", \"field\": {slot}, \"offset\": {offset} }}"
        ),
    };
    let uniform_size = (kernel.fields.len() * 4).max(16);
    let manifest = format!(
        "{{\n  \"schema\": \"heterowasm.kernel.v1\",\n  \"wgsl\": \"kernel.wgsl\",\n  \
         \"workgroup_size\": {},\n  \"dispatch\": {dispatch},\n  \
         \"uniform_size\": {uniform_size},\n  \
         \"min_offset_bytes\": {},\n  \
         \"max_constant_bytes\": {},\n  \"max_stride_bytes\": {},\n  \
         \"index_field\": {},\n  \
         \"launch_count\": {},\n  \
         \"storage_binding\": 0,\n  \"params_binding\": 1,\n  \
         \"result_slots\": {},\n  \"results_binding\": 2,\n  \
         \"fields_resolved\": {},\n  \"fields\": [\n{fields}\n  ]\n}}\n",

        kernel.workgroup_size,
        kernel.min_constant_bytes,
        kernel.max_constant_bytes,
        kernel.max_stride_bytes,
        kernel
            .index_field
            .map(|slot| i64::try_from(slot).unwrap_or(-1))
            .unwrap_or(-1),
        kernel.launch_count.map(i64::from).unwrap_or(-1),
        kernel.results.len(),
        mapping.is_some()
    );
    Ok(Artifact {
        wgsl: &kernel.source,
        manifest,
    })
}
