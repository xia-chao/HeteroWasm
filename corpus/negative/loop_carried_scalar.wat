(module
  (memory (export "mem") 16)
  (func $reduce (export "main") (param $out i32) (param $in i32) (param $n i32)
    (local $i i32) (local $s i32)
    (block $e (loop $l
      (br_if $e (i32.ge_s (local.get $i) (local.get $n)))
      (local.set $s (i32.add (local.get $s)
        (i32.load (i32.add (local.get $in) (i32.mul (local.get $i) (i32.const 4))))))
      (local.set $i (i32.add (local.get $i) (i32.const 1)))
      (br $l)))
    (i32.store (local.get $out) (local.get $s))
  )
)
