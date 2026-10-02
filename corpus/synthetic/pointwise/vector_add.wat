(module
  (memory 1)
  (func $vector_add (param $a i32) (param $b i32) (param $c i32) (param $n i32)
    (local $i i32)
    (block $exit
      (loop $loop
        (br_if $exit (i32.ge_s (local.get $i) (local.get $n)))
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
            (i32.load
              (i32.add
                (local.get $b)
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
  (export "main" (func $vector_add))
  (export "mem" (memory 0))
)
