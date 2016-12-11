#ifndef FLOWER_NET_H
#define FLOWER_NET_H

#include <flower/layer.h>
#include <flower/gradient_descent.h>
#include <vector>

namespace flower {
    template<typename Scalar>
    class Net
    {
        friend class GradientDescent<Scalar>;
    public:
        Net();
        ~Net();

        Tensor<Scalar, 2, RowMajor> infer(const Tensor<Scalar, 2, RowMajor> &data) const;

        void add_layer(const ILayer<Scalar> &layer);

        const std::vector<LayerPtr<Scalar>> &layers() const;
        inline int epoch() const { return epoch_; }

    private:
        std::vector<LayerPtr<Scalar>> layers_;
        int epoch_;
    };

    #include <flower/net.inl>
}


#endif
