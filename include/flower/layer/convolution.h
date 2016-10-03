#ifndef FLOWER_CONVOLUTION_H
#define FLOWER_CONVOLUTION_H

#include <flower/layer.h>

namespace flower
{
    class ConvolutionDef : public ILayerDef
    {
    public:
        ConvolutionDef();

        inline const char *type() const { return "Tanh"; }

    protected:
        layer_ptr create(Net *net) const;

        int filter_;
        int stride_;
        int padding_;
    };

    class Convolution : public ILayer
    {
    public:
        explicit Convolution(Net *net, const ConvolutionDef &definition);

        inline const char *type() const { return "Tanh"; }

        Eigen::Tensor<double, 2> forward(const Eigen::Tensor<double, 2> &data, bool train = false);
        Eigen::Tensor<double, 2> backward(const Eigen::Tensor<double, 2> &errors);

        Eigen::Tensor<double, 4> forward(const Eigen::Tensor<double, 4> &data, bool train = false);
        Eigen::Tensor<double, 4> backward(const Eigen::Tensor<double, 4> &errors);

        static void convolve(const Eigen::Tensor<double, 3>& input, const Eigen::Tensor<double, 3>& filter, double bias, Eigen::Tensor<double, 2>& output, int stride);

    protected:
        Eigen::Tensor<double, 3> kernel_;
        int stride_;
        int padding_;
    };
}

#endif
