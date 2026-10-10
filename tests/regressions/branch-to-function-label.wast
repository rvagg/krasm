;; A branch to a function's own label is a return. The caller must continue
;; in its own frame: here each caller uses a label after the call (`return`
;; or `br` to its own blocks), which fails if the callee's frame lingers.

(module
  (func $by-br (i32.const 3) (i64.const 1) (br 0))
  (func $by-br-if (i32.const 3) (br_if 0 (i32.const 1)) (drop))
  (func $by-br-table (i32.const 3) (br_table 0 0 (i32.const 1)) (drop))
  (func $value-by-br (result i32) (i64.const 1) (br 0 (i32.const 7)))

  (func (export "return-after-br") (result i32)
    (f32.const 0) (f64.const 0) (call $by-br) (i32.const 9) (return))
  (func (export "return-after-br-if") (result i32)
    (f32.const 0) (call $by-br-if) (drop) (i32.const 9) (return))
  (func (export "block-after-br-table") (result i32)
    (block (result i32) (call $by-br-table) (br 0 (i32.const 9))))
  (func (export "value-then-return") (result i32)
    (call $value-by-br) (return))
)
(assert_return (invoke "return-after-br") (i32.const 9))
(assert_return (invoke "return-after-br-if") (i32.const 9))
(assert_return (invoke "block-after-br-table") (i32.const 9))
(assert_return (invoke "value-then-return") (i32.const 7))
