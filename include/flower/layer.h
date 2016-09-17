#ifndef FLOWER_LAYER_H
#define FLOWER_LAYER_H

#include <Eigen/Core>
#include <memory>

namespace flower {
    class Net;
    class Feature;
    class ILayer;

    typedef std::shared_ptr<ILayer> layer_ptr;

    class ILayerDef
    {
    public:
        ILayerDef();

        virtual inline const char *type() const = 0;

        virtual layer_ptr create(Net *net, const char* name) const = 0;
    };

    class ILayer
    {
    public:
        ILayer() = delete;
        ILayer(const ILayer&) = delete;

        explicit ILayer(Net* net, const char *name, const ILayerDef &definition, unsigned int size_x, unsigned size_y);

        virtual inline const char *type() const = 0;
        virtual inline const char *name() const { return name_; }

        virtual const Eigen::MatrixXd &forward(const Eigen::MatrixXd &bottom_feat) = 0;
        virtual const Eigen::MatrixXd &backward(const Eigen::MatrixXd &top_diff) = 0;

    protected:
        const char *name_;
        Net* net_;
        Eigen::MatrixXd feat_;
        Eigen::MatrixXd diff_;
    };
}

#endif
