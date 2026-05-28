use std::fmt::{Debug, Display};

use backend_uzu::{
    ArrayElement, DataType,
    backends::{
        common::{
            Backend, Context, Encoder, Kernels,
            kernel::{ManualKernels, QuantizedMatmulQmvLloydMaxKernel, matmul::MatmulKernel},
        },
        cpu::Cpu,
    },
};
use num_traits::Float;

use super::check_tolerance;
use crate::{
    common::{
        helpers::allocation_to_vec,
        matmul::{LloydMaxQuantBuffers, LloydMaxQuantInput, lloyd_max_quant_arguments},
    },
    uzu_test,
};

fn run_kernel<B: Backend, T: ArrayElement + Float>(input: &LloydMaxQuantInput<T>) -> Vec<T> {
    let context = B::Context::new().expect("Failed to create Context");
    let mut buffers = LloydMaxQuantBuffers::<B, T>::allocate(&context, input);

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    let kernel = <<B as Backend>::Kernels as Kernels>::QuantizedMatmulQmvLloydMaxKernel::new(
        &context,
        T::data_type(),
        input.group_size,
        4,
    )
    .expect("Failed to create QuantizedMatmulQmvLloydMaxKernel");
    kernel.encode(
        &buffers.w,
        &buffers.scales,
        &buffers.codebook,
        &buffers.bias_indices,
        &buffers.bias_codebook,
        &buffers.x,
        &mut buffers.y,
        input.k,
        input.n,
        input.m,
        &mut encoder,
    );

    encoder.end_encoding().submit().wait_until_completed().expect("Failed to wait command buffer");
    allocation_to_vec(&buffers.y)
}

fn run_matmul_dispatch<B: Backend, T: ArrayElement + Float>(input: &LloydMaxQuantInput<T>) -> Vec<T> {
    let context = B::Context::new().expect("Failed to create Context");
    let mut buffers = LloydMaxQuantBuffers::<B, T>::allocate(&context, input);

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    let mut matmul = <<B as Backend>::Kernels as ManualKernels>::MatmulKernel::new(&context, T::data_type())
        .expect("Failed to create MatmulKernel");
    matmul
        .encode(lloyd_max_quant_arguments(&mut buffers, input), &mut encoder)
        .expect("Failed to encode Lloyd-Max matmul");

    encoder.end_encoding().submit().wait_until_completed().expect("Failed to wait command buffer");
    allocation_to_vec(&buffers.y)
}

fn assert_lloyd_max_gemm_unsupported<B: Backend>() {
    let input = LloydMaxQuantInput::<half::bf16>::new(5, 512, 64, 64);
    let context = B::Context::new().expect("Failed to create Context");
    let mut buffers = LloydMaxQuantBuffers::<B, half::bf16>::allocate(&context, &input);

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    let mut matmul = <<B as Backend>::Kernels as ManualKernels>::MatmulKernel::new(&context, half::bf16::data_type())
        .expect("Failed to create MatmulKernel");
    let error = matmul
        .encode(lloyd_max_quant_arguments(&mut buffers, &input), &mut encoder)
        .expect_err("Lloyd-Max GEMM should return an unsupported error");
    let message = error.to_string();
    assert!(message.contains("Unsupported matmul feature Lloyd-Max GEMM"), "unexpected error: {message}");
}

fn assert_matches_expected<T: ArrayElement + Float + Debug + Display>(
    input: &LloydMaxQuantInput<T>,
    output: &[T],
    expected: &[T],
) {
    let (rel_tol, abs_tol): (f64, f64) = match T::data_type() {
        DataType::BF16 => (0.05, 0.5),
        DataType::F16 => (0.02, 0.35),
        _ => (0.01, 0.1),
    };

    let mut errors = 0;
    for (output_idx, (&expected_value, &actual_value)) in expected.iter().zip(output.iter()).enumerate() {
        let expected_f32 = expected_value.to_f32().unwrap();
        let actual_f32 = actual_value.to_f32().unwrap();
        if !check_tolerance(expected_f32, actual_f32, rel_tol, abs_tol) {
            if errors < 5 {
                eprintln!(
                    "idx={output_idx} expected={expected_f32} actual={actual_f32} diff={}",
                    (expected_f32 - actual_f32).abs()
                );
            }
            errors += 1;
        }
    }
    assert_eq!(
        errors, 0,
        "QMV Lloyd-Max kernel: m={}, k={}, n={}, group_size={}: {errors} mismatches",
        input.m, input.k, input.n, input.group_size,
    );
}

fn test_kernel<T: ArrayElement + Float + Debug + Display>(group_size: u32) {
    for batch_size in [1usize, 2, 4] {
        for k in [512usize, 1024] {
            let input = LloydMaxQuantInput::<T>::new(batch_size, k, 64, group_size);
            let expected = run_kernel::<Cpu, T>(&input);
            for_each_non_cpu_backend!(|B| {
                let output = run_kernel::<B, T>(&input);
                assert_matches_expected(&input, &output, &expected);
            });
        }
    }
}

#[uzu_test]
fn test_qmv_lloyd_max_group_size_32() {
    for_each_float_type!(|F| {
        test_kernel::<F>(32);
    });
}

#[uzu_test]
fn test_qmv_lloyd_max_group_size_64() {
    for_each_float_type!(|F| {
        test_kernel::<F>(64);
    });
}

#[uzu_test]
fn test_qmv_lloyd_max_group_size_128() {
    for_each_float_type!(|F| {
        test_kernel::<F>(128);
    });
}

fn hand_traced_output<T: ArrayElement + Float>(
    input: &LloydMaxQuantInput<T>,
    batch: usize,
    out_row: usize,
) -> f32 {
    let k = input.k as usize;
    let n = input.n as usize;
    let gs = input.group_size as usize;
    let num_groups_k = k / gs;
    let bias_stride = num_groups_k.div_ceil(2);
    assert!(out_row < n);
    assert!(batch < input.m as usize);

    let mut acc = 0.0f32;
    for l in 0..k {
        let wl_idx = out_row * k + l;
        let u32_idx = wl_idx / 8;
        let nibble = wl_idx % 8;
        let w_code = ((input.w_packed[u32_idx] >> (nibble * 4)) & 0xF) as usize;

        let g = l / gs;
        let scale = input.scales[out_row * num_groups_k + g].to_f32().unwrap();

        let byte_idx = out_row * bias_stride + (g >> 1);
        let byte = input.bias_indices[byte_idx];
        let b_code = if g & 1 == 0 {
            (byte & 0x0F) as usize
        } else {
            ((byte >> 4) & 0x0F) as usize
        };

        let cb_val = input.codebook[w_code].to_f32();
        let bias_val = input.bias_codebook[b_code].to_f32();
        let dequant = (cb_val - bias_val) * scale;

        let x_val = input.x[batch * k + l].to_f32().unwrap();
        acc += x_val * dequant;
    }
    acc
}

#[uzu_test]
fn test_qmv_lloyd_max_hand_traced() {
    let input = LloydMaxQuantInput::<f32>::new(2, 512, 32, 64);

    let cpu = run_kernel::<Cpu, f32>(&input);
    for batch in 0..(input.m as usize) {
        for j in 0..(input.n as usize) {
            let expected = hand_traced_output(&input, batch, j);
            let actual = cpu[batch * input.n as usize + j];
            assert!(
                (expected - actual).abs() < 1e-3,
                "CPU vs hand-trace mismatch at batch={batch} j={j}: hand={expected} cpu={actual}"
            );
        }
    }

    for_each_non_cpu_backend!(|B| {
        let metal = run_kernel::<B, f32>(&input);
        for batch in 0..(input.m as usize) {
            for j in 0..(input.n as usize) {
                let expected = hand_traced_output(&input, batch, j);
                let actual = metal[batch * input.n as usize + j];
                let tol = (expected.abs() * 0.01).max(0.1);
                assert!(
                    (expected - actual).abs() <= tol,
                    "Metal vs hand-trace mismatch at batch={batch} j={j}: hand={expected} metal={actual}"
                );
            }
        }
    });
}

#[uzu_test]
fn test_qmv_lloyd_max_dispatch_path() {
    let input = LloydMaxQuantInput::<half::bf16>::new(4, 1024, 64, 64);
    let expected = run_matmul_dispatch::<Cpu, half::bf16>(&input);
    for_each_non_cpu_backend!(|B| {
        let output = run_matmul_dispatch::<B, half::bf16>(&input);
        assert_matches_expected(&input, &output, &expected);
    });
}

#[uzu_test]
fn test_qmv_lloyd_max_gemm_returns_unsupported() {
    for_each_backend!(|B| {
        assert_lloyd_max_gemm_unsupported::<B>();
    });
}
