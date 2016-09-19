#ifndef FLOWER_RELU_H
#define FLOWER_RELU_H

#include <flower/layer.h>

namespace flower
{
    class ReluDef : public ILayerDef
    {
    public:
        ReluDef();

        inline const char *type() const { return "Relu"; }

    protected:
        layer_ptr create(Net *net, const char* name) const;
    };

    class Relu : public ILayer
    {
    public:
        explicit Relu(Net* net, const char *name, const ReluDef &definition);

        inline const char *type() const { return "Relu"; }

        Eigen::MatrixXd forward(const Eigen::MatrixXd &data);
        Eigen::MatrixXd backward(const Eigen::MatrixXd &errors);

    protected:
        Eigen::MatrixXd data_;
    };
}

#endif
