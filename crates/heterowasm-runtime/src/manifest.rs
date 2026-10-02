use std::path::Path;

use crate::error::RuntimeError;
use crate::types::{DispatchSpec, FieldMapping, KernelSpec};


pub(crate) fn load_dir(dir: &Path) -> Result<Vec<KernelSpec>, RuntimeError> {
    let index_path = dir.join("kernels.json");
    let index_text = read(&index_path)?;
    let names = parse_kernel_index(&index_text, &index_path)?;
    if names.is_empty() {
        return Err(RuntimeError::BadManifest {
            path: index_path,
            detail: "kernels list is empty — no executable kernel".to_string(),
        });
    }

    names.into_iter().map(|name| load_one(dir, &name)).collect()
}

fn load_one(dir: &Path, name: &str) -> Result<KernelSpec, RuntimeError> {
    let wgsl_path = dir.join(format!("{name}.wgsl"));
    let json_path = dir.join(format!("{name}.json"));
    let source = read(&wgsl_path)?;
    let text = read(&json_path)?;
    parse_kernel(&text, &json_path, name, source)
}

fn read(path: &Path) -> Result<String, RuntimeError> {
    std::fs::read_to_string(path).map_err(|source| RuntimeError::Read {
        path: path.to_path_buf(),
        source,
    })
}


pub(crate) fn parse_kernel_index(text: &str, path: &Path) -> Result<Vec<String>, RuntimeError> {
    let list = array_after(text, "kernels").ok_or_else(|| RuntimeError::BadManifest {
        path: path.to_path_buf(),
        detail: "`kernels` array not found".to_string(),
    })?;
    let mut names = Vec::new();
    for piece in list.split(',') {
        let name = piece.trim().trim_matches('"').trim();
        if !name.is_empty() {
            names.push(name.to_string());
        }
    }
    Ok(names)
}


pub(crate) fn parse_kernel(
    text: &str,
    path: &Path,
    name: &str,
    source: String,
) -> Result<KernelSpec, RuntimeError> {
    let bad = |detail: String| RuntimeError::BadManifest {
        path: path.to_path_buf(),
        detail,
    };

    let workgroup_size = number_after(text, "workgroup_size")
        .ok_or_else(|| bad("workgroup_size not found".into()))?;
    let uniform_size =
        number_after(text, "uniform_size").ok_or_else(|| bad("uniform_size not found".into()))?;
    let dispatch = parse_dispatch(text).ok_or_else(|| bad("cannot parse dispatch".into()))?;

    let max_constant_bytes = number_after(text, "max_constant_bytes").ok_or_else(|| {
        bad("max_constant_bytes not found — it participates in upper-bound checks (re-run `heterowasm compile`)".into())
    })?;
    let max_stride_bytes = number_after(text, "max_stride_bytes").ok_or_else(|| {
        bad(
            "max_stride_bytes not found — upper-bound checks must not assume unit stride (re-run `heterowasm compile`)"
                .into(),
        )
    })?;
    let min_offset_bytes = number_after(text, "min_offset_bytes").ok_or_else(|| {
        bad("min_offset_bytes not found — it drives lower-bound checks (re-run `heterowasm compile`)".into())
    })?;


    let result_slots = number_after(text, "result_slots").unwrap_or(0);

    let fields = match array_after(text, "fields") {
        Some(list) => parse_fields(list),
        None => Vec::new(),
    };
    let fields_resolved = flag_after(text, "fields_resolved");

    Ok(KernelSpec {
        name: name.to_string(),
        source,
        workgroup_size: u32::try_from(workgroup_size)
            .map_err(|_| bad("workgroup_size out of range".into()))?,
        dispatch,
        uniform_size: usize::try_from(uniform_size)
            .map_err(|_| bad("uniform_size out of range".into()))?,
        fields,
        fields_resolved,
        result_slots: usize::try_from(result_slots)
            .map_err(|_| bad("result_slots out of range".into()))?,
        min_offset_bytes,
        max_constant_bytes,
        max_stride_bytes,
        index_field: number_after(text, "index_field")
            .and_then(|value| usize::try_from(value).ok()),
        launch_count: number_after(text, "launch_count")
            .and_then(|value| u32::try_from(value).ok()),
    })
}

fn flag_after(text: &str, key: &str) -> bool {
    let needle = format!("\"{key}\":");
    let Some(start) = text.find(&needle) else {
        return false;
    };
    text[start + needle.len()..]
        .trim_start()
        .starts_with("true")
}

fn parse_dispatch(text: &str) -> Option<DispatchSpec> {
    let start = text.find("\"dispatch\"")?;
    let rest = &text[start..];
    let kind = string_after(rest, "kind")?;
    match kind.as_str() {
        "fixed" => Some(DispatchSpec::Fixed(
            u32::try_from(number_after(rest, "count")?).ok()?,
        )),
        "from_field" => Some(DispatchSpec::FromField(
            usize::try_from(number_after(rest, "field")?).ok()?,
        )),
        "from_field_mask" => Some(DispatchSpec::FromFieldMask(
            usize::try_from(number_after(rest, "field")?).ok()?,
            u32::try_from(number_after(rest, "mask")?).ok()?,
        )),
        "from_field_mask_add" => Some(DispatchSpec::FromFieldMaskAdd(
            usize::try_from(number_after(rest, "field")?).ok()?,
            u32::try_from(number_after(rest, "mask")?).ok()?,
            u32::try_from(number_after(rest, "addend")?).ok()?,
        )),
        "from_sum" => Some(DispatchSpec::FromSum(
            usize::try_from(number_after(rest, "left")?).ok()?,
            usize::try_from(number_after(rest, "right")?).ok()?,
        )),
        "from_sum_shift" => Some(DispatchSpec::FromSumShift(
            usize::try_from(number_after(rest, "base")?).ok()?,
            usize::try_from(number_after(rest, "shifted")?).ok()?,
            u32::try_from(number_after(rest, "amount")?).ok()?,
        )),
        "from_sub_shift" => Some(DispatchSpec::FromSubShift(
            usize::try_from(number_after(rest, "base")?).ok()?,
            usize::try_from(number_after(rest, "minuend")?).ok()?,
            usize::try_from(number_after(rest, "subtrahend")?).ok()?,
            u32::try_from(number_after(rest, "amount")?).ok()?,
        )),
        "from_loaded_shift_mask" => Some(DispatchSpec::FromLoadedShiftMask(
            usize::try_from(number_after(rest, "base")?).ok()?,
            usize::try_from(number_after(rest, "shifted")?).ok()?,
            u32::try_from(number_after(rest, "amount")?).ok()?,
            u32::try_from(number_after(rest, "mask")?).ok()?,
        )),
        "from_loaded" => Some(DispatchSpec::FromLoaded(
            usize::try_from(number_after(rest, "field")?).ok()?,
            u32::try_from(number_after(rest, "offset")?).ok()?,
        )),
        _ => None,
    }
}

fn parse_fields(list: &str) -> Vec<FieldMapping> {
    let mut fields = Vec::new();
    for entry in list.split('{').skip(1) {
        let Some(end) = entry.find('}') else {
            continue;
        };
        let entry = &entry[..end];
        if let (Some(slot), Some(wasm_param)) = (
            number_after(entry, "slot"),
            number_after(entry, "wasm_param"),
        ) {
            if let (Ok(slot), Ok(wasm_param)) = (usize::try_from(slot), usize::try_from(wasm_param))
            {
                fields.push(FieldMapping { slot, wasm_param });
            }
        }
    }
    fields
}


fn string_after(text: &str, key: &str) -> Option<String> {
    let at = text.find(&format!("\"{key}\""))?;
    let rest = &text[at + key.len() + 2..];
    let colon = rest.find(':')?;
    let rest = rest[colon + 1..].trim_start();
    let rest = rest.strip_prefix('"')?;
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

fn number_after(text: &str, key: &str) -> Option<i64> {
    let at = text.find(&format!("\"{key}\""))?;
    let rest = &text[at + key.len() + 2..];
    let colon = rest.find(':')?;
    let rest = rest[colon + 1..].trim_start();
    let digits: String = rest
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '-')
        .collect();
    digits.parse().ok()
}


fn array_after<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    let at = text.find(&format!("\"{key}\""))?;
    let rest = &text[at + key.len() + 2..];
    let open = rest.find('[')?;
    let rest = &rest[open + 1..];
    let close = rest.find(']')?;
    Some(&rest[..close])
}
