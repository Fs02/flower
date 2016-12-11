/*
 * Based on: http://eric-yuan.me/cpp-read-cifar10/
 */

#ifndef FLOWER_SAMPLES_CIFAR10
#define FLOWER_SAMPLES_CIFAR10

#include <flower/tensor.h>
#include <fstream>

namespace flower {
    template<typename Scalar>
    class Cifar10
    {
    public:
        void read_batch(const char *path, Tensor<Scalar, 4, RowMajor> &images, Tensor<Scalar, 2, RowMajor> &labels);
    };

    #include <data/cifar10.inl>
}

#endif
