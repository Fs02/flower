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
template<int in_rank, int out_rank>
Tensor<Scalar, 0> GradientDescent<Scalar>::feed(Tensor<Scalar, in_rank> &data, Tensor<Scalar, out_rank> &target)
{
    // increase epoch
    net_->epoch_++;

    // forward propagate
    TensorData<Scalar> predict(data.data(), data.size());
    for(const auto &layer : net_->layers())
    {
        predict = layer->forward(predict, true);
    }

    auto predict_tensor = predict.template map<out_rank>(target.dimensions());

    // TODO
    Tensor<Scalar, 0> total_error = (target - predict_tensor).pow(2.0).mean();

    // back propagate
    array<int, 2> transpose({1, 0});
    Tensor<Scalar, out_rank> errors = -(target - predict_tensor).shuffle(transpose);
    TensorData<Scalar> errors_data(errors.data(), errors.size());

    for(auto i = net_->layers().rbegin(); i != net_->layers().rend(); ++i)
    {
        errors_data = (*i)->backward(errors_data);
    }

    return total_error;
}
