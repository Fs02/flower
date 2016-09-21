#ifndef FLOWER_DROPOUT_H
#define FLOWER_DROPOUT_H

#include <flower/layer.h>

namespace flower
{
    class DropoutDef : public ILayerDef
    {
    public:
        DropoutDef(double probability = 0.5);

        inline const char *type() const { return "Droput"; }

        inline double probability() const { return probability_; }

        double probability_;

    protected:
        layer_ptr create(Net *net, const char* name) const;
    };

    class Dropout : public ILayer
    {
    public:
        explicit Dropout(Net *net, const char *name, const DropoutDef &definition);

        inline const char *type() const { return "Droput"; }

        Eigen::MatrixXd forward(const Eigen::MatrixXd &data);
        Eigen::MatrixXd backward(const Eigen::MatrixXd &errors);

    protected:
        Eigen::ArrayXXd mask_;
        double probability_;
    };
}

#endif
