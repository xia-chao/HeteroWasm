use waffle::{FunctionBody, Operator, Value, ValueDef};


pub fn canonical_value(body: &FunctionBody, value: Value) -> Value {
    let mut current = value;

    for _ in 0..64 {
        match body.values.get(current) {
            Some(ValueDef::Alias(inner)) if *inner != current => current = *inner,
            _ => break,
        }
    }
    current
}


pub fn constant_value(body: &FunctionBody, value: Value) -> Option<i64> {
    match body.values.get(canonical_value(body, value))? {
        ValueDef::Operator(Operator::I32Const { value: raw }, _, _) => Some(i32_bits_to_i64(*raw)),
        ValueDef::Operator(Operator::I64Const { value: raw }, _, _) => i64::try_from(*raw).ok(),
        _ => None,
    }
}


pub(crate) fn i32_bits_to_i64(raw: u32) -> i64 {
    i64::from(i32::from_ne_bytes(raw.to_ne_bytes()))
}
