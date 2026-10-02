(module
  (memory 1 1)
  (func $fill_constant (param $v i32)
    (local $i i32)
    (block $exit
      (loop $loop
        (br_if $exit
          (i32.ge_s
            (local.get $i)
            (i32.const 8)
          )
        )
        (i32.store
          (i32.mul
            (local.get $i)
            (i32.const 4)
          )
          (local.get $v)
        )
        (local.set $i
          (i32.add
            (local.get $i)
            (i32.const 1)
          )
        )
        (br $loop)
      )
    )
  )
  (export "main" (func $fill_constant))
  (export "mem" (memory 0))
)
