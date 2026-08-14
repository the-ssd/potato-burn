type Embedding = [u64; 4]; // 4 * 64 = 256

type BitString = Vec<u64>;

struct CPUInerence {}

fn xor(mut a: BitString, b: &BitString) -> BitString {
    for (i, a) in a.iter_mut().enumerate() {
        *a = *a ^ b[i];
    }
    a
}

fn not(mut a: BitString) -> BitString {
    for a in a.iter_mut() {
        *a = !*a;
    }
    a
}

fn count_ones(a: &BitString) -> u32 {
    let mut count = 0;
    for a in a.iter() {
        count += a.count_ones();
    }
    count
}

fn bits(a: &BitString) -> u32 {
    a.len() as u32 * u64::BITS
}

/// Dot product
fn dot(a: BitString, b: &BitString, temp: u32, bias: u32) -> bool {
    let xnor = not(xor(a, b));
    let val = count_ones(&xnor) * temp + bias;

    if val > bits(&xnor) { true } else { false }
}

fn set_bit(a: &mut Vec<BitString>, x: usize, y: usize, bit: bool) {
    let a = &mut a[x];
    let offset = y / 64;
    let bit_index = y % 64;
    let a = &mut a[y];
    let bit = (bit as u64) << bit_index;
    *a = (*a & !(1 << bit_index)) | bit;
}

fn matmul(
    a: &Vec<BitString>,
    b_transposed: &Vec<BitString>,
    temp: u32,
    bias: u32,
) -> Vec<BitString> {
    let mut new_bitmatrix = vec![vec![0; b_transposed.len()]; a.len()];

    for (i, a) in a.iter().enumerate() {
        //let mut new_bitstring = Vec::new();
        for (j, b) in b_transposed.iter().enumerate() {
            let result = dot(a.clone(), b, temp, bias);

            set_bit(&mut new_bitmatrix, i, j, result);
            //new_bitstring.push(result);
        }

        //new_bitmatrix.push(new_bitstring);
    }
    new_bitmatrix
}
