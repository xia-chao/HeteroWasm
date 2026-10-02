use std::io::Write as _;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::types::{Event, Field, Level, Stage};


pub trait Sink: Send + Sync {
    fn record(&self, event: &Event);
}


#[derive(Debug, Default)]
pub struct StderrSink;

impl Sink for StderrSink {
    fn record(&self, event: &Event) {
        let mut line = String::new();
        write_human(event, &mut line);

        let _ = std::io::stderr().write_all(line.as_bytes());
    }
}


pub struct JsonlSink {

    writer: Mutex<Box<dyn std::io::Write + Send>>,
}

impl std::fmt::Debug for JsonlSink {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("JsonlSink")
    }
}

impl JsonlSink {

    pub fn create(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let file = std::fs::File::create(path)?;
        Ok(Self {
            writer: Mutex::new(Box::new(std::io::BufWriter::new(file))),
        })
    }


    pub fn to_writer(writer: Box<dyn std::io::Write + Send>) -> Self {
        Self {
            writer: Mutex::new(writer),
        }
    }
}

impl Sink for JsonlSink {
    fn record(&self, event: &Event) {

        let Ok(mut writer) = self.writer.lock() else {
            return;
        };
        let mut line = String::new();
        write_json(event, &mut line);
        line.push('\n');
        let _ = writer.write_all(line.as_bytes());
        let _ = writer.flush();
    }
}


#[derive(Debug, Default)]
pub struct MemorySink {
    events: Mutex<Vec<Event>>,
}

impl MemorySink {

    pub fn events(&self) -> Vec<Event> {
        match self.events.lock() {
            Ok(guard) => guard.clone(),
            Err(poisoned) => poisoned.into_inner().clone(),
        }
    }


    pub fn len(&self) -> usize {
        match self.events.lock() {
            Ok(guard) => guard.len(),
            Err(poisoned) => poisoned.into_inner().len(),
        }
    }


    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }


    pub fn contains_message(&self, needle: &str) -> bool {
        self.events()
            .iter()
            .any(|event| event.message.contains(needle))
    }


    pub fn count(&self, stage: Stage, level: Level) -> usize {
        self.events()
            .iter()
            .filter(|event| event.stage == stage && event.level == level)
            .count()
    }


    pub fn clear(&self) {
        match self.events.lock() {
            Ok(mut guard) => guard.clear(),
            Err(poisoned) => poisoned.into_inner().clear(),
        }
    }
}

impl Sink for MemorySink {
    fn record(&self, event: &Event) {
        match self.events.lock() {
            Ok(mut guard) => guard.push(event.clone()),
            Err(poisoned) => poisoned.into_inner().push(event.clone()),
        }
    }
}


pub struct Trace {
    sinks: Vec<Arc<dyn Sink>>,

    threshold: Option<Level>,
}

impl std::fmt::Debug for Trace {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Trace")
            .field("sinks", &self.sinks.len())
            .field("threshold", &self.threshold)
            .finish()
    }
}

impl Trace {

    pub fn silent() -> Self {
        Self {
            sinks: Vec::new(),
            threshold: None,
        }
    }


    pub fn with_sinks(sinks: Vec<Arc<dyn Sink>>, min_level: Level) -> Self {
        Self {
            sinks,
            threshold: Some(min_level),
        }
    }


    pub fn with_sink(sink: Arc<dyn Sink>, min_level: Level) -> Self {
        Self::with_sinks(vec![sink], min_level)
    }


    pub fn to_stderr(min_level: Level) -> Self {
        Self::with_sink(Arc::new(StderrSink), min_level)
    }


    pub fn to_jsonl(path: impl AsRef<Path>, min_level: Level) -> std::io::Result<Self> {
        Ok(Self::with_sink(
            Arc::new(JsonlSink::create(path)?),
            min_level,
        ))
    }


    pub fn share(&self) -> Self {
        Trace {
            sinks: self.sinks.iter().map(Arc::clone).collect(),
            threshold: self.threshold,
        }
    }


    pub fn to_memory(min_level: Level) -> (Self, Arc<MemorySink>) {
        let sink = Arc::new(MemorySink::default());
        let handle = Arc::clone(&sink);

        let erased: Arc<dyn Sink> = sink;
        (Self::with_sink(erased, min_level), handle)
    }


    pub fn enabled(&self, level: Level) -> bool {
        match self.threshold {
            Some(threshold) => level >= threshold,
            None => false,
        }
    }


    pub fn emit(&self, event: Event) {
        if !self.enabled(event.level) {
            return;
        }
        for sink in &self.sinks {
            sink.record(&event);
        }
    }


    pub fn log(&self, level: Level, stage: Stage, message: impl Into<String>) -> EventBuilder<'_> {
        EventBuilder {
            trace: self,
            level,
            stage,
            subject: None,
            message: message.into(),
            fields: Vec::new(),
            started: None,
        }
    }


    pub fn debug(&self, stage: Stage, message: impl Into<String>) -> EventBuilder<'_> {
        self.log(Level::Debug, stage, message)
    }


    pub fn info(&self, stage: Stage, message: impl Into<String>) -> EventBuilder<'_> {
        self.log(Level::Info, stage, message)
    }


    pub fn warn(&self, stage: Stage, message: impl Into<String>) -> EventBuilder<'_> {
        self.log(Level::Warn, stage, message)
    }


    pub fn error(&self, stage: Stage, message: impl Into<String>) -> EventBuilder<'_> {
        self.log(Level::Error, stage, message)
    }


    pub fn stage(&self, stage: Stage, subject: impl Into<String>) -> StageGuard<'_> {
        let subject = subject.into();
        self.debug(stage, "stage begin")
            .subject(subject.as_str())
            .emit();
        StageGuard {
            trace: self,
            stage,
            subject,
            started: Instant::now(),
        }
    }
}


pub struct StageGuard<'a> {
    trace: &'a Trace,
    stage: Stage,
    subject: String,
    started: Instant,
}

impl StageGuard<'_> {

    pub fn subject(&self) -> &str {
        &self.subject
    }
}

impl Drop for StageGuard<'_> {
    fn drop(&mut self) {

        self.trace
            .debug(self.stage, "stage end")
            .subject(self.subject.as_str())
            .elapsed_since(self.started)
            .emit();
    }
}


pub struct EventBuilder<'a> {
    trace: &'a Trace,
    level: Level,
    stage: Stage,
    subject: Option<String>,
    message: String,
    fields: Vec<(&'static str, Field)>,
    started: Option<Instant>,
}

impl<'a> EventBuilder<'a> {

    pub fn subject(mut self, subject: impl Into<String>) -> Self {
        self.subject = Some(subject.into());
        self
    }


    pub fn field(mut self, name: &'static str, value: impl Into<Field>) -> Self {
        self.fields.push((name, value.into()));
        self
    }


    pub fn elapsed_since(mut self, started: Instant) -> Self {
        self.started = Some(started);
        self
    }


    pub fn emit(self) {
        let elapsed = self.started.map(|started| started.elapsed());
        self.trace.emit(Event {
            level: self.level,
            stage: self.stage,
            subject: self.subject,
            message: self.message,
            fields: self.fields,
            elapsed,
        });
    }
}


#[macro_export]
macro_rules! trace_event {
    ($trace:expr, $level:expr, $stage:expr, $($arg:tt)*) => {{
        let trace = &$trace;
        if trace.enabled($level) {
            trace.log($level, $stage, format!($($arg)*)).emit();
        }
    }};
}

fn micros(duration: Duration) -> i64 {
    i64::try_from(duration.as_micros()).unwrap_or(i64::MAX)
}

fn write_human(event: &Event, out: &mut String) {
    out.push('[');
    out.push_str(event.level.label());
    out.push_str("] ");
    out.push_str(event.stage.as_str());
    if let Some(subject) = &event.subject {
        out.push(' ');
        out.push_str(subject);
    }
    out.push(' ');
    out.push_str(&event.message);
    for (name, value) in &event.fields {
        out.push(' ');
        out.push_str(name);
        out.push('=');
        write_plain_value(value, out);
    }
    if let Some(elapsed) = event.elapsed {
        out.push_str(" elapsed=");
        out.push_str(&micros(elapsed).to_string());
        out.push_str("us");
    }
    out.push('\n');
}


fn write_plain_value(value: &Field, out: &mut String) {
    match value {
        Field::Int(number) => out.push_str(&number.to_string()),
        Field::Bool(flag) => out.push_str(if *flag { "true" } else { "false" }),
        Field::Text(text) => out.push_str(text),
    }
}

fn write_json(event: &Event, out: &mut String) {
    out.push_str("{\"level\":\"");
    out.push_str(event.level.as_str());
    out.push_str("\",\"stage\":\"");
    out.push_str(event.stage.as_str());
    out.push('"');
    if let Some(subject) = &event.subject {
        out.push_str(",\"subject\":\"");
        escape_json_into(subject, out);
        out.push('"');
    }
    out.push_str(",\"message\":\"");
    escape_json_into(&event.message, out);
    out.push('"');
    if let Some(elapsed) = event.elapsed {
        out.push_str(",\"elapsed_us\":");
        out.push_str(&micros(elapsed).to_string());
    }
    if !event.fields.is_empty() {
        out.push_str(",\"fields\":{");
        for (index, (name, value)) in event.fields.iter().enumerate() {
            if index > 0 {
                out.push(',');
            }
            out.push('"');
            escape_json_into(name, out);
            out.push_str("\":");
            write_json_value(value, out);
        }
        out.push('}');
    }
    out.push('}');
}

fn write_json_value(value: &Field, out: &mut String) {
    match value {
        Field::Int(number) => out.push_str(&number.to_string()),
        Field::Bool(flag) => out.push_str(if *flag { "true" } else { "false" }),
        Field::Text(text) => {
            out.push('"');
            escape_json_into(text, out);
            out.push('"');
        }
    }
}


pub fn escape_json_into(text: &str, out: &mut String) {
    for character in text.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            control if u32::from(control) < 0x20 => {
                out.push_str("\\u");
                let code = u32::from(control);
                for shift in [12_u32, 8, 4, 0] {
                    let digit = (code >> shift) & 0xF;
                    out.push(char::from_digit(digit, 16).unwrap_or('0'));
                }
            }
            other => out.push(other),
        }
    }
}


#[cfg(test)]
mod tests;
