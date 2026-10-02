(module
  (memory 1)
  (func $two_passes (param $a i32) (param $c i32) (param $n i32)
    (local $i i32)
    (block $exit_first
      (loop $loop_first
        (br_if $exit_first (i32.ge_s (local.get $i) (local.get $n)))
        (i32.store
          (i32.add
            (local.get $c)
            (i32.mul (local.get $i) (i32.const 4))
          )
          (i32.add
            (i32.load
              (i32.add
                (local.get $a)
                (i32.mul (local.get $i) (i32.const 4))
              )
            )
            (i32.const 1)
          )
        )
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $loop_first)
      )
    )
    (local.set $i (i32.const 0))
    (block $exit_second
      (loop $loop_second
        (br_if $exit_second (i32.ge_s (local.get $i) (local.get $n)))
        (i32.store
          (i32.add
            (local.get $a)
            (i32.mul (local.get $i) (i32.const 4))
          )
          (i32.mul
            (i32.load
              (i32.add
                (local.get $c)
                (i32.mul (local.get $i) (i32.const 4))
              )
            )
            (i32.const 3)
          )
        )
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $loop_second)
      )
    )
  )
  (export "main" (func $two_passes))
  (export "mem" (memory 0))
)
