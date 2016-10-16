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
        GradientDescent(Net<Scalar> *net, double learning_rate);
        GradientDescent(Net<Scalar> *net, const IOptimizer<Scalar>& optimizer);
        ~GradientDescent();

        template<int in_rank, int out_rank>
        Tensor<Scalar, 0> feed(const Tensor<Scalar, in_rank> &data, const Tensor<Scalar, out_rank> &target);

    private:
        Net<Scalar> *net_;
    };

    #include <flower/gradient_descent.inl>
}

#endif
