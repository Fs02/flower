#ifndef FLOWER_TANH_H
#define FLOWER_TANH_H

#include <flower/layer.h>

namespace flower
{
    class TanhDef : public ILayerDef
    {
    public:
        TanhDef(unsigned int size);

        inline const char *type() const { return "Tanh"; }

        inline unsigned int size() const { return size_; }

        unsigned int size_;

    protected:
        layer_ptr create(Net *net, const char* name) const;
    };

    class Tanh : public ILayer
    {
    public:
        explicit Tanh(Net* net, const char *name, const TanhDef &definition);

        inline const char *type() const { return "Tanh"; }

        Eigen::MatrixXd forward(const Eigen::MatrixXd &data);
        Eigen::MatrixXd backward(const Eigen::MatrixXd &errors);

    protected:
        Eigen::MatrixXd data_;
    };
}

#endif