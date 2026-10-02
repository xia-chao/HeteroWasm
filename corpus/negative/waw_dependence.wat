(module
  (memory 1)
  (func $waw (param $a i32) (param $n i32) (param $v i32)
    (local $i i32)
    (block $exit
      (loop $loop
        (br_if $exit
          (i32.ge_s
            (local.get $i)
            (local.get $n)
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
          (local.get $v)
        )
        (i32.store
          (i32.add
            (local.get $a)
            (i32.mul
              (i32.sub
                (local.get $i)
                (i32.const 1)
              )
              (i32.const 4)
            )
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
)
