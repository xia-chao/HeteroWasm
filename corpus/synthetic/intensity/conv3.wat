(module
  (memory (export "mem") 16)
  (func $conv3 (export "main") (param $out i32) (param $in i32) (param $n i32)
    (local $i i32)
    (block $e (loop $l
      (br_if $e (i32.ge_s (local.get $i) (local.get $n)))
      (i32.store
        (i32.add (local.get $out) (i32.mul (i32.add (local.get $i) (i32.const 1)) (i32.const 4)))
        (i32.add (i32.add (i32.mul (i32.const 1) (i32.load (i32.add (local.get $in) (i32.mul (local.get $i) (i32.const 4))))) (i32.mul (i32.const 2) (i32.load (i32.add (local.get $in) (i32.mul (i32.add (local.get $i) (i32.const 1)) (i32.const 4)))))) (i32.mul (i32.const 3) (i32.load (i32.add (local.get $in) (i32.mul (i32.add (local.get $i) (i32.const 2)) (i32.const 4)))))))
      (local.set $i (i32.add (local.get $i) (i32.const 1)))
      (br $l)))
  )
)
