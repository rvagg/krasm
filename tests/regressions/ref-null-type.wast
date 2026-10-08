;; The `ref.null` immediate is a reference type (0x70 funcref, 0x6F externref).
;; Each module has one `[] -> []` function whose body is `ref.null t; drop`.

(module binary
  "\00asm" "\01\00\00\00"
  "\01\04\01\60\00\00"                   ;; type section: [] -> []
  "\03\02\01\00"                         ;; function section: one function, type 0
  "\0a\07\01\05\00\d0\70\1a\0b"          ;; code: ref.null func, drop, end
)

(module binary
  "\00asm" "\01\00\00\00"
  "\01\04\01\60\00\00"
  "\03\02\01\00"
  "\0a\07\01\05\00\d0\6f\1a\0b"          ;; ref.null extern
)

;; A number type is not a reference type, so these bodies do not decode.
(assert_malformed
  (module binary
    "\00asm" "\01\00\00\00"
    "\01\04\01\60\00\00"
    "\03\02\01\00"
    "\0a\07\01\05\00\d0\7f\1a\0b"        ;; ref.null i32
  )
  "malformed reference type"
)

(assert_malformed
  (module binary
    "\00asm" "\01\00\00\00"
    "\01\04\01\60\00\00"
    "\03\02\01\00"
    "\0a\07\01\05\00\d0\7b\1a\0b"        ;; ref.null v128
  )
  "malformed reference type"
)
