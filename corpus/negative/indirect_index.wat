(module
  (memory 1)
  (func $gather (param $a i32) (param $b i32) (param $idx i32) (param $n i32)
    (local $i i32)
    (block $exit
      (loop $loop
        (br_if $exit (i32.ge_s (local.get $i) (local.get $n)))
        (i32.store
          (i32.add
            (local.get $a)
            (i32.mul (local.get $i) (i32.const 4))
          )
          (i32.load
            (i32.add
              (local.get $b)
              (i32.mul
                (i32.load
                  (i32.add
                    (local.get $idx)
                    (i32.mul (local.get $i) (i32.const 4))
                  )
                )
                (i32.const 4)
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
