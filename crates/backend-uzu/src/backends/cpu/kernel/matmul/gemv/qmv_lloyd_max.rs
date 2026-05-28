use half::{bf16, f16};
use num_traits::Float;
use proc_macros::kernel;

use crate::ArrayElement;

/// CPU reference for Lloyd-Max QMV.
///
/// Storage:
/// - `weights`        : bit-packed `BITS`-wide weight codes (uint32 backing, lo bits first)
/// - `scales`         : one fp scale per group
/// - `codebook`       : `1 << BITS` halves, weight LUT
/// - `bias_indices`   : 4-bit packed bias codes, one per group (lo nibble = even group)
/// - `bias_codebook`  : 16 halves, bias LUT (bias_bits = 4)
///
/// Dequant: `weight = (codebook[w_code] - bias_codebook[b_code]) * scale`
pub fn qmv_lloyd_max<T: ArrayElement + Float>(
    weights: *const u32,
    scales: *const T,
    codebook: *const f16,
    bias_indices: *const u8,
    bias_codebook: *const f16,
    input: *const T,
    output: *mut T,
    in_vec_size: usize,
    out_vec_size: usize,
    batch_size: usize,
    group_size: usize,
) {
    let num_groups_k = in_vec_size.div_ceil(group_size);
    // bias_bits = 4 → 2 bias indices per byte → ceil(groups/2) bytes per row.
    let bias_stride = num_groups_k.div_ceil(2);
    const PACK_FACTOR: usize = 8;

    unsafe {
        for i in 0..batch_size {
            for j in 0..out_vec_size {
                let mut acc = 0.0f32;
                for l in 0..in_vec_size {
                    let weight_linear_idx = j * in_vec_size + l;
                    let u32_idx = weight_linear_idx / PACK_FACTOR;
                    let bit_offset = (weight_linear_idx % PACK_FACTOR) * 4;
                    let val_q: usize = ((weights.add(u32_idx).read_unaligned() >> bit_offset) & 0xF) as usize;

                    let group_idx = l / group_size;
                    let scale = (*scales.add(j * num_groups_k + group_idx)).to_f32().unwrap();

                    let byte_index = j * bias_stride + (group_idx >> 1);
                    let byte_val = *bias_indices.add(byte_index);
                    let bias_code = if (group_idx & 1) == 0 {
                        (byte_val & 0x0F) as usize
                    } else {
                        ((byte_val >> 4) & 0x0F) as usize
                    };

                    let code_val = (*codebook.add(val_q)).to_f32();
                    let bias_val = (*bias_codebook.add(bias_code)).to_f32();
                    let dequant = (code_val - bias_val) * scale;

                    let val_a = (*input.add(i * in_vec_size + l)).to_f32().unwrap();
                    acc += val_a * dequant;
                }
                *output.add(i * out_vec_size + j) = T::from(acc).unwrap();
            }
        }
    }
}

#[kernel(QuantizedMatmulQmvLloydMax)]
#[variants(T, f32, f16, bf16)]
#[variants(GROUP_SIZE, 32, 64, 128)]
#[variants(BITS, 4)]
pub fn quantized_matmul_qmv_lloyd_max<T: ArrayElement + Float, const GROUP_SIZE: u32, const BITS: u32>(
    weights: *const u32,
    scales: *const T,
    codebook: *const f16,
    bias_indices: *const u8,
    bias_codebook: *const f16,
    input: *const T,
    output: *mut T,
    in_vec_size: u32,
    out_vec_size: u32,
    batch_size: u32,
) {
    qmv_lloyd_max::<T>(
        weights,
        scales,
        codebook,
        bias_indices,
        bias_codebook,
        input,
        output,
        in_vec_size as usize,
        out_vec_size as usize,
        batch_size as usize,
        GROUP_SIZE as usize,
    );
}
