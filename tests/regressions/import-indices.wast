;; Imported and defined entities share an index space for each external kind.

(module $provider
  (type $value (func (result i32)))
  (memory (export "memory") 1)
  (global (export "global") i32 (i32.const 5))
  (table (export "table") 2 funcref)
  (table (export "externs") 1 externref)
  (func (export "increment") (param i32) (result i32)
    (i32.add (local.get 0) (i32.const 1)))
  (func (export "wide") (param i64) (result i64)
    (i64.add (local.get 0) (i64.const 2)))
  (func (export "check") (param i32) (result i32)
    (call_indirect (type $value) (local.get 0)))
)
(register "provider" $provider)

;; table.init works when the only table is imported. Its writes are shared.
(module $imported_table
  (import "provider" "table" (table 2 funcref))
  (func $seven (result i32) (i32.const 7))
  (elem $values func $seven)
  (func (export "init")
    (table.init $values (i32.const 1) (i32.const 0) (i32.const 1)))
)
(assert_return (invoke $imported_table "init"))
;; The provider calls $seven through the shared table, observing the importer's write.
(assert_return (invoke $provider "check" (i32.const 1)) (i32.const 7))

;; Defined tables follow imported tables, even when their reference types differ.
(module $local_table
  (import "provider" "table" (table 2 funcref))
  (table $local 1 externref)
  (elem $values externref (ref.null extern))
  (func (export "init")
    (table.init $local $values (i32.const 0) (i32.const 0) (i32.const 1))
    (elem.drop $values))
)
(assert_return (invoke $local_table "init"))
(assert_trap (invoke $local_table "init") "out of bounds table access")
(assert_return (invoke $provider "check" (i32.const 1)) (i32.const 7))

;; The reference type must come from the addressed table, not a local namesake.
(assert_invalid
  (module
    (import "provider" "externs" (table 1 externref))
    (table 1 funcref)
    (elem $values funcref (ref.null func))
    (func (table.init 0 $values (i32.const 0) (i32.const 0) (i32.const 1))))
  "type mismatch")
(assert_invalid
  (module
    (import "provider" "table" (table 2 funcref))
    (table 1 externref)
    (elem $values funcref (ref.null func))
    (func (table.init 1 $values (i32.const 0) (i32.const 0) (i32.const 1))))
  "type mismatch")
(assert_invalid
  (module
    (import "provider" "table" (table 2 funcref))
    (table 1 funcref)
    (elem $values funcref (ref.null func))
    (func (table.init 2 $values (i32.const 0) (i32.const 0) (i32.const 1))))
  "unknown table 2")

;; Function indices exclude intervening memory, global and table imports.
(module $mixed_imports
  (type $unary (func (param i32) (result i32)))
  (import "provider" "memory" (memory 1))
  (import "provider" "increment" (func $increment (param i32) (result i32)))
  (import "provider" "global" (global i32))
  (import "provider" "wide" (func $wide (param i64) (result i64)))
  (import "provider" "table" (table 2 funcref))
  (elem (i32.const 0) func $increment)
  (func (export "direct") (result i32 i64)
    (call $increment (i32.const 10)) (call $wide (i64.const 20)))
  (func (export "indirect") (result i32)
    (call_indirect (type $unary) (i32.const 30) (i32.const 0)))
)
(assert_return (invoke $mixed_imports "direct") (i32.const 11) (i64.const 22))
(assert_return (invoke $mixed_imports "indirect") (i32.const 31))
