template<typename Scalar>
Net<Scalar>::Net()
    : layers_(), epoch_(0)
{}

template<typename Scalar>
Net<Scalar>::~Net()
{
    layers_.clear();
}

template<typename Scalar>
Tensor<Scalar, 2> Net<Scalar>::infer(const Tensor<Scalar, 2> &data) const
{
    // forward propagate
    Eigen::Tensor<Scalar, 2> predict = data;
    for(const auto &layer : layers_)
    {
        predict = layer->forward(predict);
    }
    return predict;
}

template<typename Scalar>
void Net<Scalar>::add(const ILayer<Scalar> &layer)
{
    layers_.push_back(layer.create(this));
}

template<typename Scalar>
const std::vector<LayerPtr<Scalar>> &Net<Scalar>::layers() const
{
    return layers_;
}
