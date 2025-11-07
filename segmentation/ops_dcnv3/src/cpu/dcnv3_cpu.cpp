/*!
**************************************************************************************************
* MIT 开源协议
* 版权所有 (c) 2024 hxaxd
* 详见 LICENSE 文件
**************************************************************************************************
*/

#include <vector>

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>

#include "cpu/dcnv3_im2col_cpu.h"
#include "dcnv3_cpu.h"
#include "dcnv3_im2col_cpu.h"
#include <ATen/ATen.h>
#include <ATen/OpMathType.h>
#include <iostream>
#include <cmath>

at::Tensor dcnv3_cpu_forward(const at::Tensor &input, const at::Tensor &offset,
                             const at::Tensor &mask, const int kernel_h,
                             const int kernel_w, const int stride_h,
                             const int stride_w, const int pad_h,
                             const int pad_w, const int dilation_h,
                             const int dilation_w, const int group,
                             const int group_channels, const float offset_scale,
                             const int im2col_step, const int remove_center) {
    AT_ASSERTM(input.is_contiguous(), "input tensor has to be contiguous");
    AT_ASSERTM(offset.is_contiguous(), "offset tensor has to be contiguous");
    AT_ASSERTM(mask.is_contiguous(), "mask tensor has to be contiguous");
    AT_ASSERTM(input.is_cpu(), "input must be a CPU tensor");
    AT_ASSERTM(offset.is_cpu(), "offset must be a CPU tensor");
    AT_ASSERTM(mask.is_cpu(), "mask must be a CPU tensor");

    const int batch = input.size(0);
    const int height_in = input.size(1);
    const int width_in = input.size(2);
    const int channels = input.size(3);
    const int height_out =
        (height_in + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h +
        1;
    const int width_out =
        (width_in + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w +
        1;
    const int im2col_step_ = std::min(batch, im2col_step);

    AT_ASSERTM(batch % im2col_step_ == 0,
               "batch(%d) must divide im2col_step(%d)", batch, im2col_step_);
    AT_ASSERTM(
        channels == (group * group_channels),
        "Input channels and group times group channels wont match: (%d vs %d).",
        channels, group * group_channels);

    auto output =
        at::zeros({batch, height_out, width_out, group * group_channels},
                  input.options());

    const int batch_n = im2col_step_;
    auto output_n = output.view({batch / batch_n, batch_n, height_out,
                                 width_out, group * group_channels});
    
    const int num_batches = batch / im2col_step_;
    
    at::parallel_for(0, num_batches, 0, [&](int64_t start_n, int64_t end_n) {
        for (int64_t n = start_n; n < end_n; ++n) {
            auto input_slice = input.narrow(0, n * batch_n, batch_n);
            auto offset_slice = offset.narrow(0, n * batch_n, batch_n);
            auto mask_slice = mask.narrow(0, n * batch_n, batch_n);
            auto columns = output_n.select(0, n);
            
            AT_DISPATCH_FLOATING_TYPES(
                input.scalar_type(), "dcnv3_cpu_forward", [&] {
                    dcnv3::cpu::dcnv3_im2col_cpu(
                        input_slice, offset_slice, mask_slice, columns,
                        kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
                        dilation_h, dilation_w, group, group_channels,
                        batch_n, height_in, width_in, height_out, width_out,
                        offset_scale, remove_center);
                });
        }
    });

    return output;
}

std::vector<at::Tensor>
dcnv3_cpu_backward(const at::Tensor &input, const at::Tensor &offset,
                   const at::Tensor &mask, const int kernel_h,
                   const int kernel_w, const int stride_h, const int stride_w,
                   const int pad_h, const int pad_w, const int dilation_h,
                   const int dilation_w, const int group,
                   const int group_channels, const float offset_scale,
                   const at::Tensor &grad_output, const int im2col_step, const int remove_center) {

    AT_ASSERTM(input.is_contiguous(), "input tensor has to be contiguous");
    AT_ASSERTM(offset.is_contiguous(), "offset tensor has to be contiguous");
    AT_ASSERTM(mask.is_contiguous(), "mask tensor has to be contiguous");
    AT_ASSERTM(grad_output.is_contiguous(),
               "grad_output tensor has to be contiguous");
    AT_ASSERTM(input.is_cpu(), "input must be a CPU tensor");
    AT_ASSERTM(offset.is_cpu(), "offset must be a CPU tensor");
    AT_ASSERTM(mask.is_cpu(), "mask must be a CPU tensor");
    AT_ASSERTM(grad_output.is_cpu(),
               "grad_output must be a CPU tensor");

    const int batch = input.size(0);
    const int height_in = input.size(1);
    const int width_in = input.size(2);
    const int channels = input.size(3);
    const int height_out =
        (height_in + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h +
        1;
    const int width_out =
        (width_in + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w +
        1;
    const int im2col_step_ = std::min(batch, im2col_step);

    AT_ASSERTM(batch % im2col_step_ == 0,
               "batch(%d) must divide im2col_step(%d)", batch, im2col_step_);
    AT_ASSERTM(
        channels == (group * group_channels),
        "Input channels and group times group channels wont match: (%d vs %d).",
        channels, group * group_channels);

    auto grad_input = at::zeros_like(input);
    auto grad_offset = at::zeros_like(offset);
    auto grad_mask = at::zeros_like(mask);

    const int batch_n = im2col_step_;
    
    auto grad_output_n =
        grad_output.view({batch / im2col_step_, batch_n, height_out * width_out,
                          group, group_channels});

    const int num_batches = batch / im2col_step_;
    
    at::parallel_for(0, num_batches, 0, [&](int64_t start_n, int64_t end_n) {
        for (int64_t n = start_n; n < end_n; ++n) {
            auto input_slice = input.narrow(0, n * batch_n, batch_n);
            auto offset_slice = offset.narrow(0, n * batch_n, batch_n);
            auto mask_slice = mask.narrow(0, n * batch_n, batch_n);
            auto grad_output_g = grad_output_n.select(0, n);
            
            auto grad_input_slice = grad_input.narrow(0, n * batch_n, batch_n);
            auto grad_offset_slice = grad_offset.narrow(0, n * batch_n, batch_n);
            auto grad_mask_slice = grad_mask.narrow(0, n * batch_n, batch_n);
            
            AT_DISPATCH_FLOATING_TYPES(
                input.scalar_type(), "dcnv3_cpu_backward", [&] {
                    dcnv3::cpu::dcnv3_col2im_cpu(
                        grad_output_g, input_slice, offset_slice, mask_slice,
                        grad_input_slice, grad_offset_slice, grad_mask_slice,
                        kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
                        dilation_h, dilation_w, group, group_channels, batch_n,
                        height_in, width_in, height_out, width_out, offset_scale, remove_center);
                });
        }
    });

    return {grad_input, grad_offset, grad_mask};
}
