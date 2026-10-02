use super::{escape_json_into, Trace};
use crate::types::{Field, Level, Stage};

type TestResult = Result<(), Box<dyn std::error::Error>>;

#[test]
fn silent_trace_should_record_nothing() {
    let trace = Trace::silent();
    assert!(!trace.enabled(Level::Error));
    trace.error(Stage::Runtime, "must not appear").emit();
}

#[test]
fn memory_sink_should_capture_structured_fields() {
    let (trace, sink) = Trace::to_memory(Level::Debug);
    trace
        .info(Stage::Frontend, "load module")
        .subject("vector_add")
        .field("bytes", 1024_i64)
        .field("valid", true)
        .field("source", "corpus")
        .emit();

    let events = sink.events();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].subject.as_deref(), Some("vector_add"));
    assert_eq!(events[0].int_field("bytes"), Some(1024));
    assert_eq!(events[0].field("valid"), Some(&Field::Bool(true)));
    assert_eq!(
        events[0].field("source"),
        Some(&Field::Text("corpus".to_string()))
    );
}

#[test]
fn level_filter_should_drop_events_below_threshold() {
    let (trace, sink) = Trace::to_memory(Level::Warn);
    trace.debug(Stage::Runtime, "debug").emit();
    trace.info(Stage::Runtime, "info").emit();
    trace.warn(Stage::Runtime, "warn").emit();
    trace.error(Stage::Runtime, "error").emit();

    assert_eq!(sink.len(), 2);
    assert_eq!(sink.count(Stage::Runtime, Level::Warn), 1);
    assert_eq!(sink.count(Stage::Runtime, Level::Error), 1);
}

#[test]
fn stage_guard_should_record_enter_and_exit_with_elapsed_time() -> TestResult {
    let (trace, sink) = Trace::to_memory(Level::Debug);
    {
        let guard = trace.stage(Stage::ControlFlow, "vector_add");
        assert_eq!(guard.subject(), "vector_add");
    }

    let events = sink.events();
    assert_eq!(events.len(), 2, "record one enter and one leave");
    assert_eq!(events[0].message, "stage begin");
    assert_eq!(events[1].message, "stage end");
    assert!(
        events[1].elapsed.is_some(),
        "leave event must carry duration, else stage timing did nothing"
    );
    Ok(())
}

#[test]
fn stage_guard_should_record_exit_even_on_early_return() -> TestResult {
    let (trace, sink) = Trace::to_memory(Level::Debug);

    fn early_return(trace: &Trace) -> Result<(), &'static str> {
        let _guard = trace.stage(Stage::Dependence, "loop0");
        Err("early return")
    }

    assert!(early_return(&trace).is_err());
    assert_eq!(
        sink.len(),
        2,
        "early return must still leave an end event — why we use RAII instead of a handwritten pair"
    );
    Ok(())
}

#[test]
fn trace_event_macro_should_skip_formatting_when_filtered() {
    let (trace, sink) = Trace::to_memory(Level::Error);
    crate::trace_event!(trace, Level::Debug, Stage::Runtime, "value = {}", 42);
    assert!(sink.is_empty(), "filtered events must not enter the sink");
}

#[test]
fn trace_event_macro_should_format_all_arguments() {
    let (trace, sink) = Trace::to_memory(Level::Debug);
    crate::trace_event!(
        trace,
        Level::Info,
        Stage::ScalarEvolution,
        "recognized {}, stride {}",
        1,
        4
    );
    assert!(sink.contains_message("recognized 1, stride 4"));
}

#[test]
fn escape_should_handle_quotes_backslashes_and_control_characters() {
    let mut out = String::new();
    escape_json_into("a\"b\\c\nd\te\u{1}f", &mut out);
    assert_eq!(out, "a\\\"b\\\\c\\nd\\te\\u0001f");
}

#[test]
fn escape_should_leave_chinese_text_intact() {
    let mut out = String::new();
    escape_json_into("induction", &mut out);
    assert_eq!(out, "induction");
}

#[test]
fn jsonl_sink_should_write_one_object_per_line() -> TestResult {
    let path = std::env::temp_dir().join("heterowasm-trace-jsonl-test.jsonl");
    let _ = std::fs::remove_file(&path);
    {
        let trace = Trace::to_jsonl(&path, Level::Debug)?;
        trace
            .info(Stage::Frontend, "first")
            .subject("vector_add")
            .emit();
        trace.info(Stage::Frontend, "second").emit();
    }

    let content = std::fs::read_to_string(&path)?;
    let lines: Vec<&str> = content.lines().collect();
    assert_eq!(
        lines.len(),
        2,
        "two events must be written as two lines, not concatenated"
    );
    assert!(lines[0].starts_with('{') && lines[0].ends_with('}'));
    assert!(lines[0].contains("\"subject\":\"vector_add\""));
    assert!(lines[1].contains("second"));

    let _ = std::fs::remove_file(&path);
    Ok(())
}

#[test]
fn json_line_should_contain_each_structured_field() {
    let (trace, sink) = Trace::to_memory(Level::Debug);
    trace
        .warn(Stage::Dependence, "dependence unknown")
        .subject("loop0")
        .field("reads", 3_i64)
        .emit();

    let event = sink.events().into_iter().next();
    assert!(event.is_some());
    let mut line = String::new();
    if let Some(event) = event {
        super::write_json(&event, &mut line);
    }
    assert!(line.starts_with('{') && line.ends_with('}'));
    assert!(line.contains("\"level\":\"warn\""));
    assert!(line.contains("\"stage\":\"dependence\""));
    assert!(line.contains("\"subject\":\"loop0\""));
    assert!(line.contains("\"reads\":3"));
}


#[test]
fn human_format_should_end_with_newline() {
    let (trace, sink) = Trace::to_memory(Level::Debug);
    trace.info(Stage::Frontend, "first").emit();

    let mut line = String::new();
    if let Some(event) = sink.events().into_iter().next() {
        super::write_human(&event, &mut line);
    }
    assert!(
        line.ends_with('\n'),
        "human format must end with a newline, else multiple events join into one line"
    );
}
