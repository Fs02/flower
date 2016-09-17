#ifndef FLOWER_SIGMOID_H
#define FLOWER_SIGMOID_H

#include <flower/layer.h>
#include <flower/net.h>
#include <flower/feature.h>

namespace flower
{
    class SigmoidDef : public ILayerDef
    {
    public:
        SigmoidDef(unsigned int size);

        inline const char *type() const { return "Sigmoid"; }

        inline unsigned int size() const { return size_; }

        unsigned int size_;

        layer_ptr create(Net *net, const char* name) const;
    };

    class Sigmoid : public ILayer
    {
    public:
        explicit Sigmoid(Net* net, const char *name, const SigmoidDef &definition);

        inline const char *type() const { return "Sigmoid"; }

        void forward(Feature &bottom, Feature &top);
        void backward(Feature &top, Feature &bottom);

        const Eigen::MatrixXd &forward(const Eigen::MatrixXd &bottom_feat);
        const Eigen::MatrixXd &backward(const Eigen::MatrixXd &top_diff);
    };
}

#endif
