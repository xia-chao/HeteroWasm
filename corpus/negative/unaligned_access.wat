(module
  (memory 1)
  (func $byte_copy (param $a i32) (param $c i32) (param $n i32)
    (local $i i32)
    (block $exit
      (loop $loop
        (br_if $exit (i32.ge_s (local.get $i) (local.get $n)))
        (i32.store8
          (i32.add
            (local.get $c)
            (local.get $i)
          )
          (i32.load8_u
            (i32.add
              (local.get $a)
              (local.get $i)
            )
          )
        )
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $loop)
      )
    )
  )
)
