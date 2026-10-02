#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LowerError {

    NoExtent,

    NoConstantStart,

    NoConstantLimit,

    LimitTooLarge,
    NonZeroStart,

    UnsupportedOperator,

    UnsupportedFieldAffine,

    UnsupportedFieldOpaque,

    UnresolvableField,

    UnsupportedValueKind,

    MissingOperand,

    MissingValueDefinition,

    UnsupportedStoreShape,
    ExpressionTooDeep,
    UnsupportedBase,
    Unaligned,

    NothingToEmit,
}

impl LowerError {

    pub fn as_str(self) -> &'static str {
        match self {
            LowerError::NoExtent => "no_extent",
            LowerError::NoConstantStart => "no_constant_start",
            LowerError::NoConstantLimit => "no_constant_limit",
            LowerError::LimitTooLarge => "limit_too_large",
            LowerError::NonZeroStart => "non_zero_start",
            LowerError::UnsupportedOperator => "unsupported_operator",
            LowerError::UnsupportedFieldAffine => "unsupported_field_affine",
            LowerError::UnsupportedFieldOpaque => "unsupported_field_opaque",
            LowerError::UnresolvableField => "unresolvable_field",
            LowerError::UnsupportedValueKind => "unsupported_value_kind",
            LowerError::MissingOperand => "missing_operand",
            LowerError::MissingValueDefinition => "missing_value_definition",
            LowerError::UnsupportedStoreShape => "unsupported_store_shape",
            LowerError::ExpressionTooDeep => "expression_too_deep",
            LowerError::UnsupportedBase => "unsupported_base",
            LowerError::Unaligned => "unaligned",
            LowerError::NothingToEmit => "nothing_to_emit",
        }
    }
}

impl std::fmt::Display for LowerError {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        out.write_str(self.as_str())
    }
}

impl std::error::Error for LowerError {}
