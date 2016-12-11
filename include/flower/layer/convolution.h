#ifndef FLOWER_CONVOLUTION_H
#define FLOWER_CONVOLUTION_H

#include <flower/layer.h>
#include <vector>

namespace flower
{
    template<typename Scalar>
    class Convolution : public ILayer<Scalar>
    {
    public:
        Convolution(const array<int, 3>& filter_dims, int filter_num, int stride, int padding);

        inline const char *type() const { return "Convolution"; }

        inline const array<int, 3> &filter_dims() const { return filter_dims_; }
        inline int filter_num() const { return filter_num_; }
        inline int stride() const { return stride_; }
        inline int padding() const { return padding_; }

    protected:
        LayerPtr<Scalar> create(Net<Scalar> *net) const;

        Eigen::array<int, 3> filter_dims_;
        int filter_num_;
        int stride_;
        int padding_;
    };

    template<typename Scalar>
    class ConvolutionOp : public ILayerOp<Scalar>
    {
    public:
        explicit ConvolutionOp(Net<Scalar> *net, const Convolution<Scalar> &definition);

        inline const char *type() const { return "Convolution"; }

        TensorData<Scalar> forward(TensorData<Scalar> &bottom, bool train = false);
        TensorData<Scalar> backward(TensorData<Scalar> &top);

        Eigen::Tensor<double, 2> forward(const Eigen::Tensor<double, 2> &data, bool train = false);
        Eigen::Tensor<double, 2> backward(const Eigen::Tensor<double, 2> &errors);

        Eigen::Tensor<double, 4> forward(const Eigen::Tensor<double, 4> &data, bool train = false);
        Eigen::Tensor<double, 4> backward(const Eigen::Tensor<double, 4> &errors);

        static void convolve(const Eigen::Tensor<Scalar, 4>& input, Eigen::Tensor<Scalar, 3>& output, const Eigen::Tensor<Scalar, 3>& filter, double bias, int stride);

    protected:
        Tensor<double, 3> filters_;
        Tensor<double, 1> biases_;
        int stride_;
        int padding_;
    };

    #include <flower/layer/convolution.inl>
}

#endif
