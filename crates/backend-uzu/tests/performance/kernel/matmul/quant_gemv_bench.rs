use backend_uzu::{
    ArrayElement,
    backends::common::{
        Allocation, Backend, Context, Kernels,
        gpu_types::QuantizationMethod,
        kernel::{QuantizedMatmulQmvFastKernel, QuantizedMatmulQmvKernel},
    },
};
use criterion::{BenchmarkId, Criterion, Throughput};
use half::{bf16, f16};
use num_traits::Float;

use crate::{
    common::{
        matmul::{CodebookQuantBuffers, CodebookQuantInput, QuantBuffers, QuantInput, Shape, iter_encode_loop},
        type_short_name,
    },
    uzu_bench,
};

fn qmv_benchmark_shapes() -> impl Iterator<Item = Shape> {
    let matrix_shapes: &[(usize, usize)] = &[(2048, 2048), (4096, 4096), (4096, 14336), (14336, 4096), (14336, 14336)];
    let batch_sizes: &[usize] = &[1, 2, 4];
    matrix_shapes.iter().flat_map(move |&(k, n)| batch_sizes.iter().map(move |&m| Shape::new(m, k, n)))
}

fn bench_qmv_typed<B: Backend, T: ArrayElement + Float>(
    c: &mut Criterion,
    context: &B::Context,
    label: &str,
    group_size: u32,
    bits: u32,
    quant_method: QuantizationMethod,
) {
    let mut group = c.benchmark_group(format!("{}/Kernel/Qmv/{}", type_short_name::<B>(), label));

    for shape in qmv_benchmark_shapes() {
        let (m, k, n) = (shape.m, shape.k, shape.n);
        let input = QuantInput::<T>::new(m, k, n, group_size, bits, quant_method, 42);
        let mut buffers = QuantBuffers::<B, T>::allocate(context, &input);

        let kernel = <<B as Backend>::Kernels as Kernels>::QuantizedMatmulQmvKernel::new(
            context,
            T::data_type(),
            group_size,
            bits,
            quant_method,
        )
        .unwrap();

        group.throughput(Throughput::Elements((m * n * k) as u64));
        group.bench_function(BenchmarkId::from_parameter(shape.to_string()), |b| {
            iter_encode_loop::<B, _>(context, b, |encoder| {
                kernel.encode(
                    &buffers.w,
                    &buffers.scales,
                    buffers.zp.as_ref(),
                    buffers.bias.as_ref(),
                    &buffers.x,
                    &mut buffers.y,
                    k as u32,
                    n as u32,
                    m as u32,
                    encoder,
                );
            });
        });
    }
}

fn bench_qmv_fast_typed<B: Backend, T: ArrayElement + Float>(
    c: &mut Criterion,
    context: &B::Context,
    label: &str,
    group_size: u32,
    bits: u32,
    quant_method: QuantizationMethod,
) {
    let mut group = c.benchmark_group(format!("{}/Kernel/QmvFast/{}", type_short_name::<B>(), label));

    for shape in qmv_benchmark_shapes() {
        let (m, k, n) = (shape.m, shape.k, shape.n);
        let input = QuantInput::<T>::new(m, k, n, group_size, bits, quant_method, 42);
        let mut buffers = QuantBuffers::<B, T>::allocate(context, &input);

        let kernel = <<B as Backend>::Kernels as Kernels>::QuantizedMatmulQmvFastKernel::new(
            context,
            T::data_type(),
            group_size,
            bits,
            quant_method,
            false,
        )
        .unwrap();

        group.throughput(Throughput::Elements((m * n * k) as u64));
        group.bench_function(BenchmarkId::from_parameter(shape.to_string()), |b| {
            iter_encode_loop::<B, _>(context, b, |encoder| {
                kernel.encode(
                    &buffers.w,
                    &buffers.scales,
                    buffers.zp.as_ref(),
                    buffers.bias.as_ref(),
                    None::<&Allocation<B>>,
                    &buffers.x,
                    &mut buffers.y,
                    None::<&Allocation<B>>,
                    k as u32,
                    n as u32,
                    m as u32,
                    encoder,
                );
            });
        });
    }
}

fn bench_qmv_fast_codebook_typed<B: Backend, T: ArrayElement + Float>(
    c: &mut Criterion,
    context: &B::Context,
    label: &str,
    group_size: u32,
) {
    let mut group = c.benchmark_group(format!("{}/Kernel/QmvFast/{}", type_short_name::<B>(), label));

    for shape in qmv_benchmark_shapes() {
        let (m, k, n) = (shape.m, shape.k, shape.n);
        let input = CodebookQuantInput::<T>::new(m, k, n, group_size);
        let mut buffers = CodebookQuantBuffers::<B, T>::allocate(context, &input);

        let kernel = <<B as Backend>::Kernels as Kernels>::QuantizedMatmulQmvFastKernel::new(
            context,
            T::data_type(),
            group_size,
            4,
            QuantizationMethod::Codebook,
            false,
        )
        .unwrap();

        group.throughput(Throughput::Elements((m * n * k) as u64));
        group.bench_function(BenchmarkId::from_parameter(shape.to_string()), |b| {
            iter_encode_loop::<B, _>(context, b, |encoder| {
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
                    encoder,
                );
            });
        });
    }
}

#[uzu_bench]
fn bench_qmv_fast(c: &mut Criterion) {
    for_each_backend!(|B| {
        let context = <B as Backend>::Context::new().unwrap();
        bench_qmv_typed::<B, bf16>(c, &context, "ScaleBias_BF16_gs64", 64, 4, QuantizationMethod::ScaleBias);
        bench_qmv_typed::<B, bf16>(c, &context, "ZP_BF16_gs64", 64, 4, QuantizationMethod::ScaleZeroPoint);
        bench_qmv_fast_typed::<B, bf16>(c, &context, "ScaleBias_BF16_gs32", 32, 4, QuantizationMethod::ScaleBias);
        bench_qmv_fast_typed::<B, bf16>(c, &context, "ZP_BF16_gs32", 32, 4, QuantizationMethod::ScaleZeroPoint);
        bench_qmv_fast_typed::<B, bf16>(c, &context, "ScaleBias_BF16_gs64", 64, 4, QuantizationMethod::ScaleBias);
        bench_qmv_fast_typed::<B, bf16>(c, &context, "ZP_BF16_gs64", 64, 4, QuantizationMethod::ScaleZeroPoint);
        bench_qmv_fast_typed::<B, bf16>(c, &context, "ScaleBias_BF16_gs128", 128, 4, QuantizationMethod::ScaleBias);
        bench_qmv_fast_typed::<B, bf16>(c, &context, "ZP_BF16_gs128", 128, 4, QuantizationMethod::ScaleZeroPoint);
        bench_qmv_fast_typed::<B, f16>(c, &context, "ZP_F16_gs64", 64, 4, QuantizationMethod::ScaleZeroPoint);
        bench_qmv_fast_typed::<B, bf16>(c, &context, "ZP_BF16_gs64_8b", 64, 8, QuantizationMethod::ScaleZeroPoint);
        bench_qmv_fast_codebook_typed::<B, bf16>(c, &context, "NF4_BF16_gs64", 64);
        bench_qmv_fast_codebook_typed::<B, f16>(c, &context, "NF4_F16_gs64", 64);
    });
}
