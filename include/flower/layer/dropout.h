#ifndef FLOWER_DROPOUT_H
#define FLOWER_DROPOUT_H

#include <flower/layer.h>

namespace flower
{
    class Dropout : public ILayerDef
    {
    public:
        Dropout(double probability = 0.5);

        inline const char *type() const { return "Dropout"; }

        inline double probability() const { return probability_; }

        double probability_;

    protected:
        layer_ptr create(Net *net) const;
    };

    class DropoutLayer : public ILayer
    {
    public:
        explicit DropoutLayer(Net *net, const Dropout &definition);

        inline const char *type() const { return "Dropout"; }

        Tensor<double, 2, RowMajor> forward(const Tensor<double, 2, RowMajor> &data, bool train = false);
        Tensor<double, 2, RowMajor> backward(const Tensor<double, 2, RowMajor> &errors);

    protected:
        Tensor<double, 2, RowMajor> mask_;
        double probability_;
    };
}

#endif
