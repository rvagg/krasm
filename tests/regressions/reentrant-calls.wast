;; An instance waiting on an external call can be called again by its callee.
;;
;; Imports cannot be circular, so $a cannot import $b's functions. Instead $a
;; exports a table that $b fills with its own functions when instantiated.
;; Those funcrefs belong to another instance, so $a's `call_indirect` through
;; them suspends $a while the Store runs the callee, which then calls back in.
;;
;; Slots in $a's table, as filled by $b:
;;   0  $b.twice   calls $a.inc twice
;;   1  $b.leaf    doubles its argument
;;   2  $b.bounce  calls $a.down
;;   3  $b.leaky   calls $a.leaky twice

(module $a
  (type $unary (func (param i32) (result i32)))
  (table $slots (export "table") 4 funcref)
  (global $calls (mut i32) (i32.const 0))

  ;; Counts its calls, so nested activations can be observed afterwards.
  (func (export "inc") (param i32) (result i32)
    (global.set $calls (i32.add (global.get $calls) (i32.const 1)))
    (i32.add (local.get 0) (i32.const 1)))
  (func (export "calls") (result i32) (global.get $calls))

  ;; Keeps an operand beneath the outgoing call, which must survive re-entry.
  (func (export "run") (param i32) (result i32)
    (i32.add
      (i32.const 1000)
      (call_indirect $slots (type $unary) (local.get 0) (i32.const 0))))

  ;; Called by $b: suspends $a while $b, itself suspended, is re-entered.
  (func (export "inc_via_b") (param i32) (result i32)
    (i32.add (i32.const 100)
      (call_indirect $slots (type $unary) (local.get 0) (i32.const 1))))

  ;; Mutual recursion through $b: depth n crosses the instances 2n times.
  (func (export "down") (param i32) (result i32)
    (if (result i32) (i32.eqz (local.get 0))
      (then (i32.const 0))
      (else
        (i32.add (i32.const 1)
          (call_indirect $slots (type $unary)
            (i32.sub (local.get 0) (i32.const 1)) (i32.const 2))))))

  ;; Returns from beneath a leftover operand. As a re-entered entry function,
  ;; its leftover would land on the suspended activation's operands.
  (func (export "leaky") (param i32) (result i32)
    (i32.const 999) (return (i32.add (local.get 0) (i32.const 1))))
  (func (export "run_leaky") (param i32) (result i32)
    (i32.add
      (i32.const 1000)
      (call_indirect $slots (type $unary) (local.get 0) (i32.const 3))))

  (func (export "trap_inside") (param i32) (result i32) (unreachable))
)
(register "a" $a)

(module $b
  (import "a" "table" (table $slots 4 funcref))
  (import "a" "inc" (func $inc (param i32) (result i32)))
  (import "a" "inc_via_b" (func $inc_via_b (param i32) (result i32)))
  (import "a" "down" (func $down (param i32) (result i32)))
  (import "a" "leaky" (func $a_leaky (param i32) (result i32)))
  (func $twice (param i32) (result i32)
    (call $inc (call $inc (local.get 0))))
  (func $leaf (param i32) (result i32)
    (i32.mul (local.get 0) (i32.const 2)))
  (func $bounce (param i32) (result i32)
    (call $down (local.get 0)))
  (func $leaky (param i32) (result i32)
    (call $a_leaky (call $a_leaky (local.get 0))))
  ;; Fills slots 0-3 of $a's table during instantiation.
  (elem (table $slots) (i32.const 0) func $twice $leaf $bounce $leaky)
  (func (export "nested") (param i32) (result i32) (call $inc_via_b (local.get 0)))
)
(register "b" $b)

;; $a -> slot 0 -> $a.inc twice: 1000 + ((5 + 1) + 1).
(assert_return (invoke $a "run" (i32.const 5)) (i32.const 1007))
(assert_return (invoke $a "calls") (i32.const 2))

;; Each re-entered `leaky` discards its 999 before $a resumes beneath it:
;; 1000 + ((5 + 1) + 1), not 999 + 7.
(assert_return (invoke $a "run_leaky" (i32.const 5)) (i32.const 1007))

;; $b -> $a.inc_via_b -> slot 1 re-enters $b: 100 + 4 * 2. Two suspensions,
;; $b's then $a's, resume in reverse order.
(assert_return (invoke $b "nested" (i32.const 4)) (i32.const 108))

;; Recursion alternating between instances through slot 2.
(assert_return (invoke $a "down" (i32.const 50)) (i32.const 50))
(assert_exhaustion (invoke $a "down" (i32.const 100000)) "call stack exhausted")
;; Abandoned activations do not linger.
(assert_return (invoke $a "run" (i32.const 1)) (i32.const 1003))
(assert_return (invoke $a "down" (i32.const 3)) (i32.const 3))

;; A trap in a re-entered activation abandons the whole call, then $a recovers.
(module $c
  (import "a" "table" (table $slots 4 funcref))
  (import "a" "trap_inside" (func $trap (param i32) (result i32)))
  (func $call_trap (param i32) (result i32) (call $trap (local.get 0)))
  ;; Replaces slot 0, so $a.run now re-enters $a.trap_inside.
  (elem (table $slots) (i32.const 0) func $call_trap)
)
(assert_trap (invoke $a "run" (i32.const 1)) "unreachable")
(assert_return (invoke $a "calls") (i32.const 4))
(assert_return (invoke $a "inc" (i32.const 1)) (i32.const 2))
