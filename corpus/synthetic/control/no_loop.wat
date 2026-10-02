(module
  (memory 1)
  (func $store_once (param $c i32) (param $v i32)
    (i32.store
      (local.get $c)
      (local.get $v)
    )
  )
  (export "main" (func $store_once))
  (export "mem" (memory 0))
)
