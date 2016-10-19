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

        Tensor<Scalar, 2> infer(const Tensor<Scalar, 2> &data) const;

        void add_layer(const ILayer<Scalar> &layer);

        const std::vector<LayerPtr<Scalar>> &layers() const;
        inline int epoch() const { return epoch_; }

    private:
        std::vector<LayerPtr<Scalar>> layers_;
        int epoch_;
    };

    #include <flower/net.inl>
}

/*
namespace fl {
    template<typename Scalar, int rank>
    class IOperation
    {
        virtual const Eigen::Tensor<Scalar, rank> compute(const Eigen::Tensor<Scalar, rank>& input) = 0;
    };

    class ILayer
    {
    public:
        virtual std::shared_ptr<IOperation<Scalar, rank>> forward() = 0;
        virtual std::shared_ptr<IOperation<Scalar, rank>> backward() = 0;
    };

    template<typename Scalar, int rank>
    class DoublerForward : public IOperation
    {
        const Eigen::Tensor<Scalar, rank> compute(const Eigen::Tensor<Scalar, rank>& input)
        {
            return input * (Scalar)2;
        }

        const Eigen::Tensor<Scalar, rank> operator()(const Eigen::Tensor<Scalar, rank>& input) const
        {
            return compute(input);
        }
    };

    template<typename Scalar, int rank>
    class DoublerBackward : public IOperation
    {
        const Eigen::Tensor<Scalar, rank> compute(const Eigen::Tensor<Scalar, rank>& input)
        {
            return input / (Scalar)2;
        }

        const Eigen::Tensor<Scalar, rank> operator()(const Eigen::Tensor<Scalar, rank>& input) const
        {
            return compute(input);
        }
    };

    template<typename Scalar, int rank>
    class Doubler : public ILayer
    {
    public:
        Doubler()
            : rank_(rank),
              forward_(new DoublerForward<Scalar, rank>()),
              backward_(new DoublerBackward<Scalar, rank>())
        {

        }

        virtual std::shared_ptr<IOperation<Scalar, rank>> forward()
        {
            return forward_;
        }

        virtual std::shared_ptr<IOperation<Scalar, rank>> backward()
        {
            return backward_;
        }

    private:
        const unsigned int rank_;
        std::shared_ptr<IOperation<Scalar, rank>> forward_;
        std::shared_ptr<IOperation<Scalar, rank>> backward_;
    };

    class Net
    {
        void add_layer(std::shared_ptr<ILayer>& layer)
        {
            layers_.push_back(layer);
        }

        void train()
        {
            Eigen::Tensor<double, 3> data(3, 3, 3);
            data.setValues({
                                 {
                                     {-1, 0, 0},
                                     {-1,-1, 0},
                                     { 0, 0, 1}
                                 },
                                 {
                                     { 0, 0,-1},
                                     { 1,-1, 1},
                                     { 0, 0, 1}
                                 },
                                 {
                                     {-1, 1,-1},
                                     { 0,-1, 0},
                                     { 1,-1,-1}
                                 }
                             });


            for (unsigned int i = 0; i < layers_.size(); ++i)
            {
                layers_[i]->forward();
            }
        }

    private:
        std::vector<std::shared_ptr<ILayer>> layers_;
    };
}
*/


#endif
