pub mod dispatch_import {
    use std::convert::Infallible;

    use heterowasm_trace::{Stage, Trace};

    use wasm_encoder::reencode::{Error as ReencodeError, Reencode, RoundtripReencoder};
    use wasm_encoder::{EntityType, ImportSection, Module, SectionId, TypeSection, ValType};
    use wasmparser::{ImportSectionReader, Parser, TypeSectionReader};


    const IMPORT_MODULE: &str = "heterowasm";
    const IMPORT_NAME: &str = "__hw_dispatch";


    struct Shifter {

        fields: usize,
        results: usize,
        dispatch_type: u32,
        import_written: bool,
    }

    impl Shifter {

        fn write_dispatch_import(&mut self, imports: &mut ImportSection) {
            imports.import(
                IMPORT_MODULE,
                IMPORT_NAME,
                EntityType::Function(self.dispatch_type),
            );
            self.import_written = true;
        }
    }

    impl Reencode for Shifter {
        type Error = Infallible;


        fn function_index(&mut self, func: u32) -> Result<u32, ReencodeError<Infallible>> {
            Ok(func + 1)
        }

        fn parse_type_section(
            &mut self,
            types: &mut TypeSection,
            section: TypeSectionReader<'_>,
        ) -> Result<(), ReencodeError<Infallible>> {
            RoundtripReencoder.parse_type_section(types, section)?;

            self.dispatch_type = types.len();
            types.ty().function(
                vec![ValType::I32; self.fields + 1],
                vec![ValType::I32; self.results],
            );
            Ok(())
        }

        fn parse_import_section(
            &mut self,
            imports: &mut ImportSection,
            section: ImportSectionReader<'_>,
        ) -> Result<(), ReencodeError<Infallible>> {

            self.write_dispatch_import(imports);
            RoundtripReencoder.parse_import_section(imports, section)
        }

        fn intersperse_section_hook(
            &mut self,
            module: &mut Module,
            after: Option<SectionId>,
            before: Option<SectionId>,
        ) -> Result<(), ReencodeError<Infallible>> {

            if !self.import_written
                && after == Some(SectionId::Type)
                && before != Some(SectionId::Import)
            {
                let mut imports = ImportSection::new();
                self.write_dispatch_import(&mut imports);
                module.section(&imports);
            }
            RoundtripReencoder.intersperse_section_hook(module, after, before)
        }
    }


    pub fn inject_dispatch_import(
        bytes: &[u8],
        fields: usize,
        results: usize,
        trace: &Trace,
    ) -> Result<Vec<u8>, String> {
        let _scope = trace.stage(Stage::Wgsl, "inject_dispatch_import");
        let mut shifter = Shifter {
            fields,
            results,
            dispatch_type: 0,
            import_written: false,
        };
        let mut module = Module::new();
        shifter
            .parse_core_module(&mut module, Parser::new(0), bytes)
            .map_err(|err| format!("{err}"))?;
        let injected = module.finish();
        trace
            .debug(
                Stage::Wgsl,
                "injected dispatch import and renumbered function indices",
            )
            .field("fields", fields as i64)
            .field("results", results as i64)
            .field("dispatch_type", i64::from(shifter.dispatch_type))
            .field("original_bytes", bytes.len() as i64)
            .field("injected_bytes", injected.len() as i64)
            .emit();
        Ok(injected)
    }
}

pub use dispatch_import::inject_dispatch_import;
