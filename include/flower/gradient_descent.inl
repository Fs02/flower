template<typename Scalar>
GradientDescent<Scalar>::GradientDescent(Net<Scalar> *net, const IOptimizer<Scalar> &optimizer)
    : net_(net)
{
    // configure all layer
    for(const auto &layer : net_->layers())
    {
        layer->configure(optimizer);
    }
}

template<typename Scalar>
GradientDescent<Scalar>::~GradientDescent()
{}

template<typename Scalar>
Tensor<Scalar, 0, RowMajor> GradientDescent<Scalar>::feed(const Tensor<Scalar, 2, RowMajor> &data, const Tensor<Scalar, 2, RowMajor> &target)
{
    // increase epoch
    net_->epoch_++;

    // forward propagate
    Tensor<Scalar, 2, RowMajor> predict = data;
    for(const auto &layer : net_->layers())
    {
        predict = layer->forward(predict, true);
    }

    // TODO
    Tensor<Scalar, 0, RowMajor> total_error = (target - predict).pow(2.0).mean();

    // back propagate
    array<int, 2> transpose({1, 0});
    Tensor<Scalar, 2, RowMajor> errors = -(target - predict).shuffle(transpose);

    for(auto i = net_->layers().rbegin(); i != net_->layers().rend(); ++i)
    {
        errors = (*i)->backward(errors);
    }

    return total_error;
}
