#ifndef FLOWER_GRADIENT_DESCENT_H
#define FLOWER_GRADIENT_DESCENT_H

#include <flower/tensor.h>
#include <vector>

namespace flower {
    template<typename Scalar> class Net;
    template<typename Scalar> class IOptimizer;

    template<typename Scalar>
    class GradientDescent
    {
    public:
        GradientDescent(Net<Scalar> *net, const IOptimizer<Scalar>& optimizer);
        ~GradientDescent();

        Tensor<Scalar, 0, RowMajor> feed(const Tensor<Scalar, 2, RowMajor> &data, const Tensor<Scalar, 2, RowMajor> &target);

    private:
        Net<Scalar> *net_;
    };

    #include <flower/gradient_descent.inl>
}

#endif
