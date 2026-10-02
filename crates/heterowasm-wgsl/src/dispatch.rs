use crate::types::{DispatchDecision, Kernel};


pub fn decide_dispatch(kernel: &Kernel, params: &[u32], buffer_bytes: u64) -> DispatchDecision {
    let Some(required) = kernel.required_bytes(params) else {
        return DispatchDecision::CpuFallback {
            reason: "dispatch count or base unavailable; cannot compute access range".to_string(),
        };
    };
    if required <= buffer_bytes {
        DispatchDecision::Gpu
    } else {
        DispatchDecision::CpuFallback {
            reason: format!(
                "this access needs {required} bytes, buffer has only {buffer_bytes} — \
                 spec §15 requires falling back to CPU (Wasm would trap; GPU would not)"
            ),
        }
    }
}
