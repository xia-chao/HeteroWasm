(module
  (memory 1)
  (func $scale (param $x i32) (result i32)
    (i32.mul
      (local.get $x)
      (i32.const 2)
    )
  )
  (func $apply (param $a i32) (param $c i32) (param $n i32)
    (local $i i32)
    (block $exit
      (loop $loop
        (br_if $exit (i32.ge_s (local.get $i) (local.get $n)))
        (i32.store
          (i32.add
            (local.get $c)
            (i32.mul (local.get $i) (i32.const 4))
          )
          (call $scale
            (i32.load
              (i32.add
                (local.get $a)
                (i32.mul (local.get $i) (i32.const 4))
              )
            )
          )
        )
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $loop)
      )
    )
  )
)
