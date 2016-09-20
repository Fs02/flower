#ifndef FLOWER_ELU_H
#define FLOWER_ELU_H

#include <flower/layer.h>

namespace flower
{
    class EluDef : public ILayerDef
    {
    public:
        EluDef(double alpha = 1.0);

        inline const char *type() const { return "Elu"; }

        inline double alpha() const { return alpha_; }

        double alpha_;

    protected:
        layer_ptr create(Net *net, const char* name) const;
    };

    class Elu : public ILayer
    {
    public:
        explicit Elu(Net *net, const char *name, const EluDef &definition);

        inline const char *type() const { return "Elu"; }

        Eigen::MatrixXd forward(const Eigen::MatrixXd &data);
        Eigen::MatrixXd backward(const Eigen::MatrixXd &errors);

    protected:
        Eigen::MatrixXd data_;
        double alpha_;
    };
}

#endif
