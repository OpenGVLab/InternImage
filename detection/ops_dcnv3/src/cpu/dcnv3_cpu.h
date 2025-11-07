/*!
**************************************************************************************************
* MIT 开源协议
* 版权所有 (c) 2024 hxaxd
* 详见 LICENSE 文件
**************************************************************************************************
*/

#pragma once
#include <torch/extension.h>

at::Tensor dcnv3_cpu_forward(const at::Tensor &input, const at::Tensor &offset,
                             const at::Tensor &mask, const int kernel_h,
                             const int kernel_w, const int stride_h,
                             const int stride_w, const int pad_h,
                             const int pad_w, const int dilation_h,
                             const int dilation_w, const int group,
                             const int group_channels, const float offset_scale,
                             const int im2col_step, const int remove_center);

std::vector<at::Tensor>
dcnv3_cpu_backward(const at::Tensor &input, const at::Tensor &offset,
                   const at::Tensor &mask, const int kernel_h,
                   const int kernel_w, const int stride_h, const int stride_w,
                   const int pad_h, const int pad_w, const int dilation_h,
                   const int dilation_w, const int group,
                   const int group_channels, const float offset_scale,
                   const at::Tensor &grad_output, const int im2col_step, const int remove_center);
