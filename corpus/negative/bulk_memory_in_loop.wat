(module
  (memory 1)
  (func $repeated_fill (param $a i32) (param $n i32)
    (local $i i32)
    (block $exit
      (loop $loop
        (br_if $exit
          (i32.ge_s
            (local.get $i)
            (local.get $n)
          )
        )
        (memory.fill
          (local.get $a)
          (i32.const 0)
          (i32.const 4)
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
)
