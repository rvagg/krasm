;; An `if` without `else` has an implicit empty else, which passes the block's
;; parameters through as its results. The parameter and result types must
;; therefore be equal, however the then-branch ends: `unreachable` there makes
;; the then-branch type-check but cannot help the missing else.

(assert_invalid
  (module (func (param i64) (result i64)
    (if (result i64) (i64.eqz (local.get 0))
      (then (unreachable)))))
  "type mismatch"
)
(assert_invalid
  (module (func (result i32)
    (if (result i32) (i32.const 0)
      (then (i32.const 1) (return)))))
  "type mismatch"
)
(assert_invalid
  (module (func (param i32)
    (local.get 0)
    (if (param i32) (i32.const 0)
      (then (drop)))))
  "type mismatch"
)

;; Equal parameter and result types are valid without else.
(module
  (func (export "inc-if") (param i32 i32) (result i32)
    (local.get 0)
    (if (param i32) (result i32) (local.get 1)
      (then (i32.const 1) (i32.add))))
  (func (export "trap-if") (param i32)
    (if (local.get 0) (then (unreachable))))
)
(assert_return (invoke "inc-if" (i32.const 41) (i32.const 1)) (i32.const 42))
(assert_return (invoke "inc-if" (i32.const 41) (i32.const 0)) (i32.const 41))
(assert_return (invoke "trap-if" (i32.const 0)))
(assert_trap (invoke "trap-if" (i32.const 1)) "unreachable")
