(module
  (memory (export "mem") 1 1)
  (func $scale_running (param $a i32)
    (local $i i32)
    (local $p i32)
    (local.set $p (local.get $a))
    (block $exit
      (loop $loop
        (br_if $exit (i32.ge_s (local.get $i) (i32.const 8)))
        (i32.store
          (local.get $p)
          (i32.mul (i32.load (local.get $p)) (i32.const 2))
        )
        (local.set $p (i32.add (local.get $p) (i32.const 4)))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $loop)
      )
    )
    (i32.store (i32.const 512) (local.get $p))
  )
  (export "main" (func $scale_running))
)
