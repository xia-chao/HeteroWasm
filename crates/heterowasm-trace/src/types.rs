use std::time::Duration;


#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Level {
    Debug,
    Info,
    Warn,
    Error,
}

impl Level {

    pub fn as_str(self) -> &'static str {
        match self {
            Level::Debug => "debug",
            Level::Info => "info",
            Level::Warn => "warn",
            Level::Error => "error",
        }
    }


    pub(crate) fn label(self) -> &'static str {
        match self {
            Level::Debug => "DEBUG",
            Level::Info => "INFO ",
            Level::Warn => "WARN ",
            Level::Error => "ERROR",
        }
    }
}


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Frontend,
    ControlFlow,
    ScalarEvolution,
    AddressRecovery,
    Dependence,
    Legality,
    Bounds,
    Wgsl,
    Runtime,
}

impl Stage {

    pub fn as_str(self) -> &'static str {
        match self {
            Stage::Frontend => "frontend",
            Stage::ControlFlow => "cfg",
            Stage::ScalarEvolution => "scev",
            Stage::AddressRecovery => "address",
            Stage::Dependence => "dependence",
            Stage::Legality => "legality",
            Stage::Bounds => "bounds",
            Stage::Wgsl => "wgsl",
            Stage::Runtime => "runtime",
        }
    }
}


#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Field {
    Int(i64),
    Bool(bool),
    Text(String),
}

impl From<i64> for Field {
    fn from(value: i64) -> Self {
        Field::Int(value)
    }
}

impl From<usize> for Field {
    fn from(value: usize) -> Self {
        Field::Int(i64::try_from(value).unwrap_or(i64::MAX))
    }
}

impl From<bool> for Field {
    fn from(value: bool) -> Self {
        Field::Bool(value)
    }
}

impl From<&str> for Field {
    fn from(value: &str) -> Self {
        Field::Text(value.to_string())
    }
}

impl From<String> for Field {
    fn from(value: String) -> Self {
        Field::Text(value)
    }
}


#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Event {
    pub level: Level,
    pub stage: Stage,
    pub subject: Option<String>,
    pub message: String,
    pub fields: Vec<(&'static str, Field)>,
    pub elapsed: Option<Duration>,
}

impl Event {

    pub fn field(&self, name: &str) -> Option<&Field> {
        self.fields
            .iter()
            .find(|(key, _)| *key == name)
            .map(|(_, value)| value)
    }


    pub fn int_field(&self, name: &str) -> Option<i64> {
        match self.field(name) {
            Some(Field::Int(value)) => Some(*value),
            _ => None,
        }
    }
}
