#ifndef FLOWER_FULLY_CONNECTED_H
#define FLOWER_FULLY_CONNECTED_H

#include <flower/layer.h>
#include <flower/optimizer.h>

namespace flower
{
    template<typename Scalar>
    class FullyConnected : public ILayer<Scalar>
    {
    public:
        FullyConnected(unsigned int input_size, unsigned int output_size);

        inline const char *type() const { return "FullyConnected"; }

        inline unsigned int input_size() const { return input_size_; }
        inline unsigned int output_size() const { return output_size_; }

    protected:
        LayerPtr<Scalar> create(Net<Scalar> *net) const;

        unsigned int input_size_;
        unsigned int output_size_;
    };

    template<typename Scalar>
    class FullyConnectedOp : public ILayerOp<Scalar>
    {
    public:
        FullyConnectedOp(Net<Scalar> *net, const FullyConnected<Scalar> &definition);

        inline const char *type() const { return "FullyConnected"; }

        void configure(const IOptimizer<Scalar> &optimizer);

        Tensor<Scalar, 2, RowMajor> forward(const Tensor<Scalar, 2, RowMajor> &bottom, bool train = false);
        Tensor<Scalar, 2, RowMajor> backward(const Tensor<Scalar, 2, RowMajor> &top);

        Tensor<Scalar, 2, RowMajor> &weights();

    protected:
        Tensor<Scalar, 2, RowMajor> data_;
        Tensor<Scalar, 2, RowMajor> weights_;

        OptimizerPtr<Scalar> optimizer_;
        FullyConnected<Scalar> definition_;
    };

    #include <flower/layer/fully_connected.inl>
}

#endif
