#ifndef FLOWER_CONVOLUTION_H
#define FLOWER_CONVOLUTION_H

#include <flower/layer.h>
#include <vector>

namespace flower
{
    class Convolution : public ILayerDef
    {
    public:
        Convolution(const Eigen::array<int, 3>& filter_dims, int filter_num, int stride, int padding);

        inline const char *type() const { return "Tanh"; }

        inline const Eigen::array<int, 3> &filter_dims() const { return filter_dims_; }
        inline int filter_num() const { return filter_num_; }
        inline int stride() const { return stride_; }
        inline int padding() const { return padding_; }

    protected:
        layer_ptr create(Net *net) const;

        Eigen::array<int, 3> filter_dims_;
        int filter_num_;
        int stride_;
        int padding_;
    };

    class ConvolutionLayer : public ILayer
    {
    public:
        explicit ConvolutionLayer(Net *net, const Convolution &definition);

        inline const char *type() const { return "Tanh"; }

        Eigen::Tensor<double, 2> forward(const Eigen::Tensor<double, 2> &data, bool train = false);
        Eigen::Tensor<double, 2> backward(const Eigen::Tensor<double, 2> &errors);

        Eigen::Tensor<double, 4> forward(const Eigen::Tensor<double, 4> &data, bool train = false);
        Eigen::Tensor<double, 4> backward(const Eigen::Tensor<double, 4> &errors);

        static void convolve(const Eigen::Tensor<double, 4>& input, Eigen::Tensor<double, 3>& output, const Eigen::Tensor<double, 3>& filter, double bias, int stride);

    protected:
        std::vector<Eigen::Tensor<double, 3>> filters_;
        std::vector<double> biases_;
        int stride_;
        int padding_;
    };
}

#endif
