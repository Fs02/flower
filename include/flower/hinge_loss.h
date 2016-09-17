#ifndef FLOWER_SVM_LOSS_H
#define FLOWER_SVM_LOSS_H

#include <flower/layer.h>
#include <flower/feature.h>

namespace flower
{
    class HingeLossDef : public ILayerDef
    {
    public:
        HingeLossDef(double regularization);

        inline const char *type() const { return "SvmLoss"; }
        inline bool regularization() const { return regularization_; }

        layer_ptr create(Net *net, const char* name) const;

    protected:
        double regularization_;
    };

    class HingeLoss : public ILayer
    {
    public:
        HingeLoss(Net *net, const char *name, const HingeLossDef &definition);

        inline const char *type() const { return "SvmLoss"; }

        void forward(Feature &bottom, Feature &top);
        void backward(Feature &top, Feature &bottom);

        const Eigen::MatrixXd &forward(const Eigen::MatrixXd &bottom_feat);
        const Eigen::MatrixXd &backward(const Eigen::MatrixXd &top_diff);

    protected:
        double regularization_;
    };
}

#endif
