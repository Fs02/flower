#ifndef FLOWER_SIGMOID_H
#define FLOWER_SIGMOID_H

#include <flower/layer.h>

namespace flower
{
    class SigmoidDef : public ILayerDef
    {
    public:
        SigmoidDef(unsigned int size);

        inline const char *type() const { return "Sigmoid"; }

        inline unsigned int size() const { return size_; }

        unsigned int size_;

    protected:
        layer_ptr create(Net *net, const char* name) const;
    };

    class Sigmoid : public ILayer
    {
    public:
        explicit Sigmoid(Net* net, const char *name, const SigmoidDef &definition);

        inline const char *type() const { return "Sigmoid"; }

        Eigen::MatrixXd forward(const Eigen::MatrixXd &data);
        Eigen::MatrixXd backward(const Eigen::MatrixXd &errors);

    protected:
        Eigen::MatrixXd data_;
    };
}

#endif
