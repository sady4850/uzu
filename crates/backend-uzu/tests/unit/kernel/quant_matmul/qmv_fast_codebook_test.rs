use std::fmt::{Debug, Display};

use backend_uzu::{
    ArrayElement, DataType,
    backends::{
        common::{
            Allocation, Backend, Context, Encoder, Kernels,
            gpu_types::QuantizationMethod,
            kernel::{ManualKernels, QuantizedMatmulQmvFastKernel, matmul::MatmulKernel},
        },
        cpu::Cpu,
    },
};
use num_traits::Float;

use super::check_tolerance;
use crate::{
    common::{
        helpers::allocation_to_vec,
        matmul::{CodebookQuantBuffers, CodebookQuantInput, codebook_quant_arguments},
    },
    uzu_test,
};

fn run_kernel<B: Backend, T: ArrayElement + Float>(input: &CodebookQuantInput<T>) -> Vec<T> {
    let context = B::Context::new().expect("Failed to create Context");
    let mut buffers = CodebookQuantBuffers::<B, T>::allocate(&context, input);

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    let kernel = <<B as Backend>::Kernels as Kernels>::QuantizedMatmulQmvFastKernel::new(
        &context,
        T::data_type(),
        input.group_size,
        4,
        QuantizationMethod::Codebook,
        false,
    )
    .expect("Failed to create QuantizedMatmulQmvFastKernel");
    kernel.encode(
        &buffers.w,
        &buffers.scales,
        None::<&Allocation<B>>,
        None::<&Allocation<B>>,
        Some(&buffers.codebook),
        &buffers.x,
        &mut buffers.y,
        None::<&Allocation<B>>,
        input.k,
        input.n,
        input.m,
        &mut encoder,
    );

    encoder.end_encoding().submit().wait_until_completed().expect("Failed to wait command buffer");
    allocation_to_vec(&buffers.y)
}

fn run_matmul_dispatch<B: Backend, T: ArrayElement + Float>(input: &CodebookQuantInput<T>) -> Vec<T> {
    let context = B::Context::new().expect("Failed to create Context");
    let mut buffers = CodebookQuantBuffers::<B, T>::allocate(&context, input);

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    let mut matmul = <<B as Backend>::Kernels as ManualKernels>::MatmulKernel::new(&context, T::data_type())
        .expect("Failed to create MatmulKernel");
    matmul
        .encode(codebook_quant_arguments(&mut buffers, input), &mut encoder)
        .expect("Failed to encode codebook matmul");

    encoder.end_encoding().submit().wait_until_completed().expect("Failed to wait command buffer");
    allocation_to_vec(&buffers.y)
}

fn assert_codebook_gemm_unsupported<B: Backend>() {
    let input = CodebookQuantInput::<half::bf16>::new(5, 512, 64, 64);
    let context = B::Context::new().expect("Failed to create Context");
    let mut buffers = CodebookQuantBuffers::<B, half::bf16>::allocate(&context, &input);

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    let mut matmul = <<B as Backend>::Kernels as ManualKernels>::MatmulKernel::new(&context, half::bf16::data_type())
        .expect("Failed to create MatmulKernel");
    let error = matmul
        .encode(codebook_quant_arguments(&mut buffers, &input), &mut encoder)
        .expect_err("codebook GEMM should return an unsupported error");
    let message = error.to_string();
    assert!(message.contains("Unsupported matmul feature codebook GEMM"), "unexpected error: {message}");
}

fn assert_matches_expected<T: ArrayElement + Float + Debug + Display>(
    input: &CodebookQuantInput<T>,
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
        "QMV fast codebook kernel: m={}, k={}, n={}, group_size={}: {errors} mismatches",
        input.m, input.k, input.n, input.group_size,
    );
}

fn test_kernel<T: ArrayElement + Float + Debug + Display>(group_size: u32) {
    for batch_size in [1usize, 2, 4] {
        for k in [512usize, 1024] {
            let input = CodebookQuantInput::<T>::new(batch_size, k, 64, group_size);
            let expected = run_kernel::<Cpu, T>(&input);
            for_each_non_cpu_backend!(|B| {
                let output = run_kernel::<B, T>(&input);
                assert_matches_expected(&input, &output, &expected);
            });
        }
    }
}

#[uzu_test]
fn test_qmv_fast_codebook_group_size_32() {
    for_each_float_type!(|F| {
        test_kernel::<F>(32);
    });
}

#[uzu_test]
fn test_qmv_fast_codebook_group_size_64() {
    for_each_float_type!(|F| {
        test_kernel::<F>(64);
    });
}

#[uzu_test]
fn test_qmv_fast_codebook_group_size_128() {
    for_each_float_type!(|F| {
        test_kernel::<F>(128);
    });
}

#[uzu_test]
fn test_qmv_fast_codebook_dispatch_path() {
    let input = CodebookQuantInput::<half::bf16>::new(4, 1024, 64, 64);
    let expected = run_matmul_dispatch::<Cpu, half::bf16>(&input);
    for_each_non_cpu_backend!(|B| {
        let output = run_matmul_dispatch::<B, half::bf16>(&input);
        assert_matches_expected(&input, &output, &expected);
    });
}

#[uzu_test]
fn test_qmv_fast_codebook_gemm_returns_unsupported() {
    for_each_backend!(|B| {
        assert_codebook_gemm_unsupported::<B>();
    });
}
