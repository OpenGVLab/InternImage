/*
**************************************************************************************************
* MIT 开源协议
* 版权所有 (c) 2024 hxaxd
* 详见 LICENSE 文件
**************************************************************************************************
*/

#pragma once

#include <ATen/ATen.h>
#include <ATen/OpMathType.h>

namespace dcnv3 {
    namespace cpu {
        template <typename scalar_t>
        using opmath_t = double;
        
        template <typename scalar_t>
        scalar_t dcnv3_im2col_bilinear(const scalar_t* bottom_data,
                                        const int& height, const int& width,
                                        const int& group,
                                        const int& group_channels,
                                        const double& h, const double& w,
                                        const int& g, const int& c);

        template <typename scalar_t>
        void dcnv3_col2im_bilinear(
            const scalar_t* bottom_data, const int& height, const int& width,
            const int& nheads, const int& group_channels, const double& h,
            const double& w, const int& m, const int& c, const double offset_scale,
            const double& top_grad, const double& mask, double* grad_im,
            double* grad_offset, double* grad_mask);

        template <typename scalar_t>
        void dcnv3_col2im_bilinear_gm(
            const scalar_t* bottom_data, const int& height, const int& width,
            const int& nheads, const int& group_channels, const double& h,
            const double& w, const int& m, const int& c, const double offset_scale,
            const double& top_grad, const double& mask, double* grad_im,
            double* grad_offset, double* grad_mask);

        void dcnv3_im2col_cpu(
            const at::Tensor& data_im, const at::Tensor& data_offset,
            const at::Tensor& data_mask, at::Tensor& data_col,
            const int kernel_h, const int kernel_w,
            const int stride_h, const int stride_w,
            const int pad_h, const int pad_w,
            const int dilation_h, const int dilation_w,
            const int group, const int group_channels,
            const int batch_n, const int height_in, const int width_in,
            const int height_out, const int width_out,
            const double offset_scale, const int remove_center);

        void dcnv3_col2im_cpu(
            const at::Tensor& grad_col, const at::Tensor& data_im,
            const at::Tensor& data_offset, const at::Tensor& data_mask,
            at::Tensor& grad_im, at::Tensor& grad_offset, at::Tensor& grad_mask,
            const int kernel_h, const int kernel_w,
            const int stride_h, const int stride_w,
            const int pad_h, const int pad_w,
            const int dilation_h, const int dilation_w,
            const int group, const int group_channels,
            const int batch_n, const int height_in, const int width_in,
            const int height_out, const int width_out,
            const double offset_scale, const int remove_center);
    }
}