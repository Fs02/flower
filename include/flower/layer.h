#ifndef FLOWER_LAYER_H
#define FLOWER_LAYER_H

#include <flower/optimizer.h>
#include <Eigen/CXX11/Tensor>
#include <memory>

namespace flower {
    class Net;
    class ILayer;
    class IOptimizerDef;

    typedef std::shared_ptr<ILayer> layer_ptr;

    class ILayerDef
    {
        friend class Net;
    public:
        ILayerDef();

        virtual inline const char *type() const = 0;

    protected:
        virtual layer_ptr create(Net *net) const = 0;
    };

    class ILayer
    {
    public:
        ILayer() = delete;
        ILayer(const ILayer&) = delete;

        explicit ILayer(Net* net, const ILayerDef &definition);

        virtual inline const char *type() const = 0;

        virtual void configure(const IOptimizerDef &optimizer_def);

        virtual Eigen::Tensor<double, 2> forward(const Eigen::Tensor<double, 2> &data, bool train = false) = 0;
        virtual Eigen::Tensor<double, 2> backward(const Eigen::Tensor<double, 2> &errors) = 0;

    protected:
        Net* net_;
    };
}

#endif
