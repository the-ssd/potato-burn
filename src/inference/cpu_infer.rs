type Embedding = [u64; 4]; // 4 * 64 = 256
struct CPUInerence {}

/// same as (a.matmul(b.transpose()) * temp + bias).sign()
fn matmul(a: u64, b: u64, temp: u32, bias: u32) -> bool {
    let xnor = !(a ^ b);
    let val = xnor.count_ones() * temp + bias;
    let z = 0i128;

    if val > u64::BITS / 2 { true } else { false }
}
