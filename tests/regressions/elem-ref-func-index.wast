;; Element segments written as expressions (flags 4-7) hold `ref.func` indices
;; that must name a function, like the index-vector forms (flags 0-3).
;; Each module has one function, a one-entry funcref table and one segment:
;; `ref.func 0` is valid, `ref.func 91` is not.

(module binary
  "\00asm" "\01\00\00\00"
  "\01\04\01\60\00\00"                   ;; type section: [] -> []
  "\03\02\01\00"                         ;; function section: one function
  "\04\04\01\70\00\01"                   ;; table section: funcref, min 1
  "\09\09\01\04\41\00\0b\01\d2\00\0b"    ;; flags 4: active, table 0, offset 0
  "\0a\04\01\02\00\0b"                   ;; code section: empty body
)
(assert_invalid
  (module binary
    "\00asm" "\01\00\00\00"
    "\01\04\01\60\00\00" "\03\02\01\00" "\04\04\01\70\00\01"
    "\09\09\01\04\41\00\0b\01\d2\5b\0b"
    "\0a\04\01\02\00\0b"
  )
  "unknown function"
)

(module binary
  "\00asm" "\01\00\00\00"
  "\01\04\01\60\00\00" "\03\02\01\00" "\04\04\01\70\00\01"
  "\09\07\01\05\70\01\d2\00\0b"          ;; flags 5: passive
  "\0a\04\01\02\00\0b"
)
(assert_invalid
  (module binary
    "\00asm" "\01\00\00\00"
    "\01\04\01\60\00\00" "\03\02\01\00" "\04\04\01\70\00\01"
    "\09\07\01\05\70\01\d2\5b\0b"
    "\0a\04\01\02\00\0b"
  )
  "unknown function"
)

(module binary
  "\00asm" "\01\00\00\00"
  "\01\04\01\60\00\00" "\03\02\01\00" "\04\04\01\70\00\01"
  "\09\0b\01\06\00\41\00\0b\70\01\d2\00\0b"  ;; flags 6: active, explicit table 0
  "\0a\04\01\02\00\0b"
)
(assert_invalid
  (module binary
    "\00asm" "\01\00\00\00"
    "\01\04\01\60\00\00" "\03\02\01\00" "\04\04\01\70\00\01"
    "\09\0b\01\06\00\41\00\0b\70\01\d2\5b\0b"
    "\0a\04\01\02\00\0b"
  )
  "unknown function"
)

(module binary
  "\00asm" "\01\00\00\00"
  "\01\04\01\60\00\00" "\03\02\01\00" "\04\04\01\70\00\01"
  "\09\07\01\07\70\01\d2\00\0b"          ;; flags 7: declarative
  "\0a\04\01\02\00\0b"
)
(assert_invalid
  (module binary
    "\00asm" "\01\00\00\00"
    "\01\04\01\60\00\00" "\03\02\01\00" "\04\04\01\70\00\01"
    "\09\07\01\07\70\01\d2\5b\0b"
    "\0a\04\01\02\00\0b"
  )
  "unknown function"
)
