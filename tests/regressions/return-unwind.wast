;; `return` and branches to the function label discard every operand beneath
;; the results. A caller holding values below the call observes any leak:
;; an invocation's results alone cannot, because only they are popped.

(module
  (func $return (result i32)
    (i32.const 100) (return (i32.const 2)))
  (func $br (result i32)
    (i32.const 100) (br 0 (i32.const 2)))
  (func $br_if (result i32)
    (i32.const 100) (br_if 0 (i32.const 2) (i32.const 1)) (drop) (drop) (i32.const 9))
  (func $br_table (result i32)
    (i32.const 100) (br_table 0 0 (i32.const 2) (i32.const 1)))

  ;; Leftovers of other types in every enclosing frame. Each `unreachable`
  ;; keeps the frame valid; execution returns before reaching it.
  (func $nested (result i32)
    (i64.const 100)
    (block (result i32)
      (f32.const 1)
      (loop (result i32)
        (f64.const 2)
        (if (result i32) (i32.const 1)
          (then (i64.const 3) (return (i32.const 2)))
          (else (i32.const 0)))
        (unreachable))
      (unreachable))
    (unreachable))

  ;; Multiple results keep their order.
  (func $pair (result i32 i64)
    (i32.const 100) (i64.const 101) (return (i32.const 2) (i64.const 3)))

  ;; No results: everything is discarded.
  (func $empty
    (i32.const 100) (i64.const 101) (return))

  ;; Parameters and locals are not on the operand stack.
  (func $param (param i32) (result i32)
    (i32.const 100) (return (local.get 0)))

  (func (export "return") (result i32) (i32.add (i32.const 5) (call $return)))
  (func (export "br") (result i32) (i32.add (i32.const 5) (call $br)))
  (func (export "br_if") (result i32) (i32.add (i32.const 5) (call $br_if)))
  (func (export "br_table") (result i32) (i32.add (i32.const 5) (call $br_table)))
  (func (export "nested") (result i32) (i32.add (i32.const 5) (call $nested)))
  (func (export "pair") (result i32)
    (i32.const 5)
    (call $pair)
    (i32.wrap_i64)
    (i32.add)
    (i32.add))
  (func (export "empty") (result i32) (i32.const 5) (call $empty))
  (func (export "param") (result i32) (i32.add (i32.const 5) (call $param (i32.const 2))))

  ;; The same through call_indirect.
  (type $result (func (result i32)))
  (table 2 funcref)
  (elem (i32.const 0) $return $nested)
  (func (export "indirect") (param i32) (result i32)
    (i32.add (i32.const 5) (call_indirect (type $result) (local.get 0))))
)

(assert_return (invoke "return") (i32.const 7))
(assert_return (invoke "br") (i32.const 7))
(assert_return (invoke "br_if") (i32.const 7))
(assert_return (invoke "br_table") (i32.const 7))
(assert_return (invoke "nested") (i32.const 7))
(assert_return (invoke "pair") (i32.const 10))
(assert_return (invoke "empty") (i32.const 5))
(assert_return (invoke "param") (i32.const 7))
(assert_return (invoke "indirect" (i32.const 0)) (i32.const 7))
(assert_return (invoke "indirect" (i32.const 1)) (i32.const 7))

;; A re-entered entry function's leftovers: see reentrant-calls.wast.
