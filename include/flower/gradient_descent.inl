template<typename Scalar>
GradientDescent<Scalar>::GradientDescent(Net<Scalar> *net, double learning_rate)
    : GradientDescent(net, Vanilla(learning_rate))
{}

template<typename Scalar>
GradientDescen<Scalar>t::GradientDescent(Net<Scalar> *net, const IOptimizer<Scalar> &optimizer)
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
template<int in_rank, int out_rank>
Tensor<Scalar, 0> GradientDescent<Scalar>::feed(const Tensor<Scalar, in_rank> &data, const Tensor<Scalar, out_rank> &target)
{
    // increase epoch
    net_->epoch_++;

    // forward propagate
    TensorData<double> predict = data;
    for(const auto &layer : net_->layers())
    {
        predict = layer->forward(predict, true);
    }

    // TODO
    Tensor<Scalar, 0> total_error = (target - predict).pow(2.0).mean();

    // back propagate
    array<int, 2> transpose({1, 0});
    Tensor<Scalar, out_rank> errors = -(target - predict).shuffle(transpose);

    for(auto i = net_->layers().rbegin(); i != net_->layers().rend(); ++i)
    {
        errors = (*i)->backward(errors);
    }

    return total_error;
}
