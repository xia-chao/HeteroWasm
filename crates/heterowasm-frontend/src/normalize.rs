pub mod bulk_memory {
    use std::borrow::Cow;
    use std::convert::Infallible;

    use heterowasm_trace::{Stage, Trace};
    use wasm_encoder::reencode::{Error as ReencodeError, Reencode, RoundtripReencoder};
    use wasm_encoder::{Instruction, Module};
    use wasmparser::{Operator, Parser};


    #[derive(Default)]
    struct Rewriter {
        rewrites: usize,
    }

    impl Reencode for Rewriter {
        type Error = Infallible;

        fn instruction<'a>(
            &mut self,
            arg: Operator<'a>,
        ) -> Result<Instruction<'a>, ReencodeError<Infallible>> {
            match arg {

                Operator::MemoryInit { mem, .. } => {
                    self.rewrites += 1;
                    Ok(Instruction::MemoryCopy {
                        src_mem: mem,
                        dst_mem: mem,
                    })
                }

                Operator::DataDrop { .. } => {
                    self.rewrites += 1;
                    Ok(Instruction::Nop)
                }

                other => RoundtripReencoder.instruction(other),
            }
        }
    }


    pub fn normalize_bulk_memory<'a>(bytes: &'a [u8], trace: &Trace) -> Cow<'a, [u8]> {
        let _scope = trace.stage(Stage::Frontend, "normalize_bulk_memory");
        let mut rewriter = Rewriter::default();
        let mut module = Module::new();
        if let Err(err) = rewriter.parse_core_module(&mut module, Parser::new(0), bytes) {

            trace
                .debug(Stage::Frontend, "normalize skipped, using original bytes")
                .field("reason", format!("{err}"))
                .field("bytes", bytes.len() as i64)
                .emit();
            return Cow::Borrowed(bytes);
        }

        if rewriter.rewrites == 0 {
            trace
                .debug(Stage::Frontend, "no bulk-memory rewrite needed")
                .field("rewrites", 0i64)
                .emit();
            return Cow::Borrowed(bytes);
        }

        let normalized = module.finish();
        trace
            .warn(Stage::Frontend, "rewrote unsupported bulk-memory operators")
            .field("rewrites", rewriter.rewrites as i64)
            .field("original_bytes", bytes.len() as i64)
            .field("normalized_bytes", normalized.len() as i64)
            .emit();
        Cow::Owned(normalized)
    }
}

pub use bulk_memory::normalize_bulk_memory;
