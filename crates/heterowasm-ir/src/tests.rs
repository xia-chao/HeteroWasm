use super::Verdict;

#[test]
fn permits_gpu_transform_should_allow_only_proven_legal() {
    assert!(Verdict::ProvenLegal.permits_gpu_transform());
}

#[test]
fn permits_gpu_transform_should_reject_unknown() {
    assert!(!Verdict::Unknown.permits_gpu_transform());
}

#[test]
fn permits_gpu_transform_should_reject_proven_illegal() {
    assert!(!Verdict::ProvenIllegal.permits_gpu_transform());
}
