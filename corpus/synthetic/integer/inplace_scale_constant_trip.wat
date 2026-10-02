(module
  (memory 1)
  (func $inplace_scale (param $a i32) (param $n i32)
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
          (i32.add
            (local.get $a)
            (i32.mul
              (local.get $i)
              (i32.const 4)
            )
          )
          (i32.mul
            (i32.load
              (i32.add
                (local.get $a)
                (i32.mul
                  (local.get $i)
                  (i32.const 4)
                )
              )
            )
            (i32.const 2)
          )
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
  (export "main" (func $inplace_scale))
  (export "mem" (memory 0))
)
