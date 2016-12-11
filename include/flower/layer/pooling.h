#ifndef FLOWER_POOLING_H
#define FLOWER_POOLING_H

#include <flower/layer.h>
#include <vector>

namespace flower
{
    class Pooling : public ILayerDef
    {
    public:
        enum Mode { Max, Avg };

        Pooling(Mode mode, const Eigen::array<int, 2> &size, int stride);

        inline const char *type() const { return "Pooling"; }

        inline Mode mode() const { return mode_; }
        inline const Eigen::array<int, 2> &size() const { return size_; }
        inline int stride() const { return stride_; }

    protected:
        layer_ptr create(Net *net) const;

        Mode mode_;
        Eigen::array<int, 2> size_;
        int stride_;
    };

    class PoolingLayer : public ILayer
    {
    public:
        explicit PoolingLayer(Net *net, const Pooling &definition);

        inline const char *type() const { return "Pooling"; }

        Eigen::Tensor<double, 2> forward(const Eigen::Tensor<double, 2> &data, bool train = false);
        Eigen::Tensor<double, 2> backward(const Eigen::Tensor<double, 2> &errors);

        Eigen::Tensor<double, 4> forward(const Eigen::Tensor<double, 4> &data, bool train = false);
        Eigen::Tensor<double, 4> backward(const Eigen::Tensor<double, 4> &errors);

        static void pool(Pooling::Mode mode, const Eigen::Tensor<double, 4> &input, Eigen::Tensor<double, 4> &output, const Eigen::array<int, 2> &size, int stride);

    protected:
        Pooling::Mode mode_;
        Eigen::array<int, 2> size_;
        int stride_;
    };
}

#endif
