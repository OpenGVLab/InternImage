/*!
**************************************************************************************************
* MIT 开源协议
* 版权所有 (c) 2024 hxaxd
* 优化版本：高精度 opmath + Inline Bilinear（无硬编码展开，避免 bug）
**************************************************************************************************
*/

#include "dcnv3_im2col_cpu.h"

#include <ATen/ATen.h>
#include <ATen/OpMathType.h>
#include <torch/types.h>
#include <cmath>
#include <iostream>

namespace dcnv3 {
    namespace cpu {

        template <typename scalar_t, typename opmath_t>
        inline opmath_t dcnv3_im2col_bilinear(
            const scalar_t* bottom_data,
            const int& height, const int& width,
            const int& group, const int& group_channels,
            const opmath_t& h, const opmath_t& w,
            const int& g, const int& c
        ) {
            const int h_low = static_cast<int>(std::floor(h));
            const int w_low = static_cast<int>(std::floor(w));
            const int h_high = h_low + 1;
            const int w_high = w_low + 1;

            const opmath_t lh = h - h_low;
            const opmath_t lw = w - w_low;
            const opmath_t hh = opmath_t(1) - lh;
            const opmath_t hw = opmath_t(1) - lw;

            const int w_stride = group * group_channels;
            const int h_stride = width * w_stride;
            const int h_low_ptr_offset = h_low * h_stride;
            const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
            const int w_low_ptr_offset = w_low * w_stride;
            const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
            const int base_ptr = g * group_channels + c;

            scalar_t v1 = scalar_t(0);
            if (h_low >= 0 && w_low >= 0) {
                v1 = bottom_data[h_low_ptr_offset + w_low_ptr_offset + base_ptr];
            }
            scalar_t v2 = scalar_t(0);
            if (h_low >= 0 && w_high <= width - 1) {
                v2 = bottom_data[h_low_ptr_offset + w_high_ptr_offset + base_ptr];
            }
            scalar_t v3 = scalar_t(0);
            if (h_high <= height - 1 && w_low >= 0) {
                v3 = bottom_data[h_high_ptr_offset + w_low_ptr_offset + base_ptr];
            }
            scalar_t v4 = scalar_t(0);
            if (h_high <= height - 1 && w_high <= width - 1) {
                v4 = bottom_data[h_high_ptr_offset + w_high_ptr_offset + base_ptr];
            }

            const opmath_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
            return w1 * static_cast<opmath_t>(v1) + w2 * static_cast<opmath_t>(v2) +
                   w3 * static_cast<opmath_t>(v3) + w4 * static_cast<opmath_t>(v4);
        }

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
            const double offset_scale, const int remove_center
        ) {
            const int num_kernels = data_col.numel();

            AT_DISPATCH_FLOATING_TYPES(data_im.scalar_type(), "dcnv3_im2col_cpu", [&] {
                using opmath_t = at::opmath_type<scalar_t>;

                const scalar_t* data_im_ptr = data_im.data_ptr<scalar_t>();
                const scalar_t* data_offset_ptr = data_offset.data_ptr<scalar_t>();
                const scalar_t* data_mask_ptr = data_mask.data_ptr<scalar_t>();
                scalar_t* data_col_ptr = data_col.data_ptr<scalar_t>();

                const int input_size = height_in * width_in;
                const int kernel_size = kernel_h * kernel_w - remove_center;
                const int qid_stride = group * group_channels;
                const int center_h = kernel_h / 2;
                const int center_w = kernel_w / 2;

                for (int index = 0; index < num_kernels; ++index) {
                    int _temp = index;
                    const int c_col = _temp % group_channels;
                    _temp /= group_channels;
                    const int sampling_index = _temp;
                    const int g_col = _temp % group;
                    _temp /= group;
                    const int p0_w = ((dilation_w * (kernel_w - 1)) >> 1) - pad_w +
                                     (_temp % width_out) * stride_w;
                    _temp /= width_out;
                    const int p0_h = ((dilation_h * (kernel_h - 1)) >> 1) - pad_h +
                                     (_temp % height_out) * stride_h;
                    _temp /= height_out;
                    const int b_col = _temp;

                    scalar_t* data_col_ptr_current = data_col_ptr + index;
                    int data_weight_ptr = sampling_index * kernel_size;
                    int data_loc_w_ptr = data_weight_ptr << 1;
                    const scalar_t* data_im_ptr_current = data_im_ptr + b_col * input_size * qid_stride;

                    const opmath_t p0_w_ = static_cast<opmath_t>(p0_w) -
                                           ((dilation_w * (kernel_w - 1)) >> 1) * offset_scale;
                    const opmath_t p0_h_ = static_cast<opmath_t>(p0_h) -
                                           ((dilation_h * (kernel_h - 1)) >> 1) * offset_scale;

                    opmath_t col = opmath_t(0);

                    for (int i = 0; i < kernel_w; ++i) {
                        for (int j = 0; j < kernel_h; ++j) {
                            if (i == center_w && j == center_h && remove_center) {
                                continue;
                            }
                            const opmath_t offset_w = static_cast<opmath_t>(data_offset_ptr[data_loc_w_ptr]);
                            const opmath_t offset_h = static_cast<opmath_t>(data_offset_ptr[data_loc_w_ptr + 1]);
                            const opmath_t loc_w = p0_w_ + (static_cast<opmath_t>(i) * dilation_w + offset_w) * offset_scale;
                            const opmath_t loc_h = p0_h_ + (static_cast<opmath_t>(j) * dilation_h + offset_h) * offset_scale;
                            const opmath_t weight = static_cast<opmath_t>(data_mask_ptr[data_weight_ptr]);

                            if (loc_h > -1 && loc_w > -1 && loc_h < height_in && loc_w < width_in) {
                                col += dcnv3_im2col_bilinear<scalar_t, opmath_t>(
                                    data_im_ptr_current, height_in, width_in, group,
                                    group_channels, loc_h, loc_w, g_col, c_col) * weight;
                            }
                            data_weight_ptr += 1;
                            data_loc_w_ptr += 2;
                        }
                    }

                    *data_col_ptr_current = static_cast<scalar_t>(col);
                }
            });
        }

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
            const double offset_scale, const int remove_center
        ) {
            const int num_kernels = grad_col.numel();

            AT_DISPATCH_FLOATING_TYPES(grad_col.scalar_type(), "dcnv3_col2im_cpu", [&] {
                using opmath_t = at::opmath_type<scalar_t>;

                const scalar_t* grad_col_ptr = grad_col.data_ptr<scalar_t>();
                const scalar_t* data_im_ptr = data_im.data_ptr<scalar_t>();
                const scalar_t* data_offset_ptr = data_offset.data_ptr<scalar_t>();
                const scalar_t* data_mask_ptr = data_mask.data_ptr<scalar_t>();

                scalar_t* grad_im_ptr = grad_im.data_ptr<scalar_t>();
                scalar_t* grad_offset_ptr = grad_offset.data_ptr<scalar_t>();
                scalar_t* grad_mask_ptr = grad_mask.data_ptr<scalar_t>();

                const int input_size = height_in * width_in;
                const int kernel_size = kernel_h * kernel_w - remove_center;
                const int qid_stride = group * group_channels;
                const int center_h = kernel_h / 2;
                const int center_w = kernel_w / 2;

                for (int index = 0; index < num_kernels; ++index) {
                    int _temp = index;
                    const int c_col = _temp % group_channels;
                    _temp /= group_channels;
                    const int sampling_index = _temp;
                    const int g_col = _temp % group;
                    _temp /= group;
                    const int p0_w = ((dilation_w * (kernel_w - 1)) >> 1) - pad_w +
                                     (_temp % width_out) * stride_w;
                    _temp /= width_out;
                    const int p0_h = ((dilation_h * (kernel_h - 1)) >> 1) - pad_h +
                                     (_temp % height_out) * stride_h;
                    _temp /= height_out;
                    const int b_col = _temp;

                    const opmath_t top_grad = static_cast<opmath_t>(grad_col_ptr[index]);
                    int data_weight_ptr = sampling_index * kernel_size;
                    int data_loc_w_ptr = data_weight_ptr << 1;
                    const int im_ptr_offset = b_col * input_size * qid_stride;
                    const scalar_t* data_im_ptr_current = data_im_ptr + im_ptr_offset;
                    scalar_t* grad_im_ptr_current = grad_im_ptr + im_ptr_offset;

                    const opmath_t p0_w_ = static_cast<opmath_t>(p0_w) -
                                           ((dilation_w * (kernel_w - 1)) >> 1) * offset_scale;
                    const opmath_t p0_h_ = static_cast<opmath_t>(p0_h) -
                                           ((dilation_h * (kernel_h - 1)) >> 1) * offset_scale;

                    for (int i = 0; i < kernel_w; ++i) {
                        for (int j = 0; j < kernel_h; ++j) {
                            if (i == center_w && j == center_h && remove_center) {
                                continue;
                            }

                            const opmath_t offset_w = static_cast<opmath_t>(data_offset_ptr[data_loc_w_ptr]);
                            const opmath_t offset_h = static_cast<opmath_t>(data_offset_ptr[data_loc_w_ptr + 1]);
                            const opmath_t loc_w = p0_w_ + (static_cast<opmath_t>(i) * dilation_w + offset_w) * offset_scale;
                            const opmath_t loc_h = p0_h_ + (static_cast<opmath_t>(j) * dilation_h + offset_h) * offset_scale;
                            const opmath_t weight = static_cast<opmath_t>(data_mask_ptr[data_weight_ptr]);

                            if (loc_h > -1 && loc_w > -1 && loc_h < height_in && loc_w < width_in) {
                                const int h_low = static_cast<int>(std::floor(loc_h));
                                const int w_low = static_cast<int>(std::floor(loc_w));
                                const int h_high = h_low + 1;
                                const int w_high = w_low + 1;

                                const opmath_t lh = loc_h - h_low;
                                const opmath_t lw = loc_w - w_low;
                                const opmath_t hh = opmath_t(1) - lh;
                                const opmath_t hw_ = opmath_t(1) - lw;

                                const int w_stride = group * group_channels;
                                const int h_stride = width_in * w_stride;
                                const int h_low_ptr_offset = h_low * h_stride;
                                const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
                                const int w_low_ptr_offset = w_low * w_stride;
                                const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
                                const int base_ptr = g_col * group_channels + c_col;

                                const opmath_t w1 = hh * hw_;
                                const opmath_t w2 = hh * lw;
                                const opmath_t w3 = lh * hw_;
                                const opmath_t w4 = lh * lw;
                                const opmath_t top_grad_im = top_grad * weight;

                                opmath_t grad_h_weight = opmath_t(0);
                                opmath_t grad_w_weight = opmath_t(0);

                                scalar_t v1 = scalar_t(0);
                                if (h_low >= 0 && w_low >= 0) {
                                    const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
                                    v1 = data_im_ptr_current[ptr1];
                                    grad_h_weight -= hw_ * static_cast<opmath_t>(v1);
                                    grad_w_weight -= hh * static_cast<opmath_t>(v1);
                                    grad_im_ptr_current[ptr1] += static_cast<scalar_t>(w1 * top_grad_im);
                                }
                                scalar_t v2 = scalar_t(0);
                                if (h_low >= 0 && w_high <= width_in - 1) {
                                    const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
                                    v2 = data_im_ptr_current[ptr2];
                                    grad_h_weight -= lw * static_cast<opmath_t>(v2);
                                    grad_w_weight += hh * static_cast<opmath_t>(v2);
                                    grad_im_ptr_current[ptr2] += static_cast<scalar_t>(w2 * top_grad_im);
                                }
                                scalar_t v3 = scalar_t(0);
                                if (h_high <= height_in - 1 && w_low >= 0) {
                                    const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
                                    v3 = data_im_ptr_current[ptr3];
                                    grad_h_weight += hw_ * static_cast<opmath_t>(v3);
                                    grad_w_weight -= lh * static_cast<opmath_t>(v3);
                                    grad_im_ptr_current[ptr3] += static_cast<scalar_t>(w3 * top_grad_im);
                                }
                                scalar_t v4 = scalar_t(0);
                                if (h_high <= height_in - 1 && w_high <= width_in - 1) {
                                    const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
                                    v4 = data_im_ptr_current[ptr4];
                                    grad_h_weight += lw * static_cast<opmath_t>(v4);
                                    grad_w_weight += lh * static_cast<opmath_t>(v4);
                                    grad_im_ptr_current[ptr4] += static_cast<scalar_t>(w4 * top_grad_im);
                                }

                                const opmath_t val = w1 * static_cast<opmath_t>(v1) + w2 * static_cast<opmath_t>(v2) +
                                                     w3 * static_cast<opmath_t>(v3) + w4 * static_cast<opmath_t>(v4);

                                grad_mask_ptr[data_weight_ptr] += static_cast<scalar_t>(top_grad * val);
                                grad_offset_ptr[data_loc_w_ptr] += static_cast<scalar_t>(offset_scale * grad_w_weight * top_grad_im);
                                grad_offset_ptr[data_loc_w_ptr + 1] += static_cast<scalar_t>(offset_scale * grad_h_weight * top_grad_im);
                            }
                            data_weight_ptr += 1;
                            data_loc_w_ptr += 2;
                        }
                    }
                }
            });
        }
    }
}
