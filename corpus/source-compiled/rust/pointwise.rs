#![no_std]

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}

#[no_mangle]
pub extern "C" fn vector_add(a: *const i32, b: *const i32, c: *mut i32, n: usize) {
    let mut i = 0usize;
    while i < n {
        unsafe { *c.add(i) = *a.add(i) + *b.add(i) };
        i += 1;
    }
}

#[no_mangle]
pub extern "C" fn saxpy(alpha: i32, a: *const i32, b: *const i32, c: *mut i32, n: usize) {
    let mut i = 0usize;
    while i < n {
        unsafe { *c.add(i) = alpha * *a.add(i) + *b.add(i) };
        i += 1;
    }
}

#[no_mangle]
pub extern "C" fn elementwise_mul(a: *const i32, b: *const i32, c: *mut i32, n: usize) {
    let mut i = 0usize;
    while i < n {
        unsafe { *c.add(i) = *a.add(i) * *b.add(i) };
        i += 1;
    }
}
