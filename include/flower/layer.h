#ifndef FLOWER_LAYER_H
#define FLOWER_LAYER_H

#include <Eigen/Core>
#include <memory>

namespace flower {
    class Net;
    class ILayer;

    typedef std::shared_ptr<ILayer> layer_ptr;

    class ILayerDef
    {
        friend class Net;
    public:
        ILayerDef();

        virtual inline const char *type() const = 0;

    protected:
        virtual layer_ptr create(Net *net, const char* name) const = 0;
    };

    class ILayer
    {
    public:
        ILayer() = delete;
        ILayer(const ILayer&) = delete;

        explicit ILayer(Net* net, const char *name, const ILayerDef &definition);

        virtual inline const char *type() const = 0;
        virtual inline const char *name() const { return name_; }

        virtual Eigen::MatrixXd forward(const Eigen::MatrixXd &data) = 0;
        virtual Eigen::MatrixXd backward(const Eigen::MatrixXd &errors) = 0;

    protected:
        const char *name_;
        Net* net_;
    };
}

#endif
