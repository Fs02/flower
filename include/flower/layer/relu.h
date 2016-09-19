#ifndef FLOWER_RELU_H
#define FLOWER_RELU_H

#include <flower/layer.h>

namespace flower
{
    class ReluDef : public ILayerDef
    {
    public:
        ReluDef(unsigned int size);

        inline const char *type() const { return "Relu"; }

        inline unsigned int size() const { return size_; }

        unsigned int size_;

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
