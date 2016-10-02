#ifndef FLOWER_FULLY_CONNECTED_H
#define FLOWER_FULLY_CONNECTED_H

#include <flower/layer.h>
#include <flower/optimizer.h>

namespace flower
{
    class FullyConnected : public ILayerDef
    {
    public:
        FullyConnected(unsigned int input_size, unsigned int output_size);

        inline const char *type() const { return "FullyConnected"; }

        inline unsigned int input_size() const { return input_size_; }
        inline unsigned int output_size() const { return output_size_; }

    protected:
        layer_ptr create(Net *net) const;

        unsigned int input_size_;
        unsigned int output_size_;
    };

    class FullyConnectedLayer : public ILayer
    {
    public:
        FullyConnectedLayer(Net *net, const FullyConnected &definition);

        inline const char *type() const { return "FullyConnected"; }

        void configure(const IOptimizerDef &optimizer_def);

        Eigen::Tensor<double, 2> forward(const Eigen::Tensor<double, 2> &data, bool train = false);
        Eigen::Tensor<double, 2> backward(const Eigen::Tensor<double, 2> &errors);

    protected:
        Eigen::Tensor<double, 2> data_;
        Eigen::Tensor<double, 2> weights_;

        optimizer_ptr optimizer_;
    };
}

#endif
