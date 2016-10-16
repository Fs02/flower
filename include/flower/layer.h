#ifndef FLOWER_LAYER_H
#define FLOWER_LAYER_H

#include <flower/tensor.h>
#include <flower/optimizer.h>
#include <memory>

namespace flower {
    template<typename Scalar> class Net;
    template<typename Scalar> class IOptimizer;
    template<typename Scalar> class ILayer;

    template<typename Scalar>
    using LayerPtr = std::shared_ptr<ILayerOp<Scalar>>;

    template<typename Scalar>
    class ILayer
    {
        friend class Net<Scalar>;
    public:
        ILayer();

        virtual inline const char *type() const = 0;

    protected:
        virtual LayerPtr<Scalar> create(Net<Scalar> *net) const = 0;
    };

    template<typename Scalar>
    class ILayerOp
    {
    public:
        ILayerOp() = delete;
        ILayerOp(const ILayerOp&) = delete;

        explicit ILayerOp(Net<Scalar>* net, const ILayer<Scalar> &definition);

        inline const char *type() const;
        inline const ILayer<Scalar>& definition();

        virtual void configure(const IOptimizerDef &optimizer_def);

        virtual TensorData<Scalar> forward(const TensorData<Scalar> &bottom, bool train = false) = 0;
        virtual TensorData<Scalar> backward(const TensorData<Scalar> &top) = 0;

    protected:
        Net<Scalar>* net_;
        ILayer<typename Scalar> definition_;
    };

    #include <flower/layer.inl>
}

#endif
