#ifndef FLOWER_LAYER_H
#define FLOWER_LAYER_H

#include <flower/tensor.h>
#include <flower/optimizer.h>
#include <memory>

namespace flower {
    template<typename Scalar> class Net;
    template<typename Scalar> class IOptimizer;
    template<typename Scalar> class ILayer;
    template<typename Scalar> class ILayerOp;

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

        explicit ILayerOp(Net<Scalar> *net, const ILayer<Scalar> &definition);

        virtual inline const char *type() const = 0;

        virtual void configure(const IOptimizer<Scalar> &optimizer);

        virtual Tensor<Scalar, 2, RowMajor> forward(const Tensor<Scalar, 2, RowMajor> &bottom, bool train = false) = 0;
        virtual Tensor<Scalar, 2, RowMajor> backward(const Tensor<Scalar, 2, RowMajor> &top) = 0;

    protected:
        Net<Scalar> *net_;
    };

    #include <flower/layer.inl>
}

#endif
