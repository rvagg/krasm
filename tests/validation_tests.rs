//! The binary decoder validates as it reads; `ValidatedModule::new` validates a
//! finished `Module`. Both guard instantiation, so they must agree: every module
//! the decoder accepts must also pass module validation.

use base64::{Engine as _, engine::general_purpose};
use krasm::ValidatedModule;
use krasm::parser::{self, reader::Reader};
use std::collections::HashMap;
use std::fs;

#[test]
fn module_validation_accepts_every_decoded_spec_module() {
    let mut fixtures: Vec<_> = fs::read_dir("tests/spec")
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "json"))
        .collect();
    fixtures.sort();
    assert!(!fixtures.is_empty());

    let mut accepted = 0;
    let mut disagreements = Vec::new();
    for path in &fixtures {
        let data: serde_json::Value = serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
        for (name, encoded) in data["bin"].as_object().unwrap() {
            let bytes = general_purpose::STANDARD.decode(encoded.as_str().unwrap()).unwrap();
            let Ok(decoded) = parser::parse(&HashMap::new(), name, &mut Reader::new(bytes)) else {
                continue;
            };
            let module = decoded.into_inner().expect("a freshly parsed module is unshared");
            match ValidatedModule::new(module) {
                Ok(_) => accepted += 1,
                Err(error) => disagreements.push(format!("{}: {name}: {error}", path.display())),
            }
        }
    }
    assert!(
        disagreements.is_empty(),
        "{} disagreements:\n{}",
        disagreements.len(),
        disagreements.join("\n")
    );
    assert!(accepted > 1000, "only {accepted} modules decoded");
}

#[test]
fn module_validation_limits_declared_locals() {
    use krasm::ValidationError;
    use krasm::parser::module::{Locals, ValueType};

    let mut module = krasm::wat::parse("(module (func))").unwrap().into_inner().unwrap();
    module.code.code[0].locals = Locals::new(vec![(u32::MAX, ValueType::I32)]);
    assert!(matches!(
        ValidatedModule::new(module),
        Err(ValidationError::TooManyLocals)
    ));
}
