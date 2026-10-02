(module
  (memory 1)
  (func $matrix_double (param $a i32) (param $rows i32) (param $cols i32)
    (local $r i32)
    (local $c i32)
    (local $offset i32)
    (block $exit_rows
      (loop $loop_rows
        (br_if $exit_rows (i32.ge_s (local.get $r) (local.get $rows)))
        (local.set $c (i32.const 0))
        (block $exit_cols
          (loop $loop_cols
            (br_if $exit_cols (i32.ge_s (local.get $c) (local.get $cols)))
            (local.set $offset
              (i32.mul
                (i32.add
                  (i32.mul (local.get $r) (local.get $cols))
                  (local.get $c)
                )
                (i32.const 4)
              )
            )
            (i32.store
              (i32.add (local.get $a) (local.get $offset))
              (i32.mul
                (i32.load
                  (i32.add (local.get $a) (local.get $offset))
                )
                (i32.const 2)
              )
            )
            (local.set $c (i32.add (local.get $c) (i32.const 1)))
            (br $loop_cols)
          )
        )
        (local.set $r (i32.add (local.get $r) (i32.const 1)))
        (br $loop_rows)
      )
    )
  )
  (export "main" (func $matrix_double))
  (export "mem" (memory 0))
)
