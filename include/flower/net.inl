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
Tensor<Scalar, 2, RowMajor> Net<Scalar>::infer(const Tensor<Scalar, 2, RowMajor> &data) const
{
    // forward propagate
    Tensor<Scalar, 2, RowMajor> predict = data;
    for(const auto &layer : layers_)
    {
        predict = layer->forward(predict);
    }
    return predict;
}

template<typename Scalar>
void Net<Scalar>::add_layer(const ILayer<Scalar> &layer)
{
    layers_.push_back(layer.create(this));
}

template<typename Scalar>
const std::vector<LayerPtr<Scalar>> &Net<Scalar>::layers() const
{
    return layers_;
}
