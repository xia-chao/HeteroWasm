(module
  (memory 1)
  (global $slot (mut i32) (i32.const 0))
  (func $spilled_carried_raw (param $a i32) (param $n i32)
    (i32.store
      (global.get $slot)
      (i32.const 0)
    )
    (block $exit
      (loop $loop
        (br_if $exit
          (i32.ge_s
            (i32.load (global.get $slot))
            (local.get $n)
          )
        )
        (i32.store
          (i32.add
            (local.get $a)
            (i32.mul
              (i32.load (global.get $slot))
              (i32.const 4)
            )
          )
          (i32.add
            (i32.load
              (i32.add
                (local.get $a)
                (i32.mul
                  (i32.sub
                    (i32.load (global.get $slot))
                    (i32.const 1)
                  )
                  (i32.const 4)
                )
              )
            )
            (i32.const 1)
          )
        )
        (i32.store
          (global.get $slot)
          (i32.add
            (i32.load (global.get $slot))
            (i32.const 1)
          )
        )
        (br $loop)
      )
    )
  )
)
