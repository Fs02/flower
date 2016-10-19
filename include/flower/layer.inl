template<typename Scalar>
ILayer<Scalar>::ILayer()
{}

template<typename Scalar>
ILayerOp<Scalar>::ILayerOp(Net<Scalar>* net, const ILayer<Scalar> &definition)
    : net_(net)
{}

template<typename Scalar>
void ILayerOp<Scalar>::configure(const IOptimizer<Scalar> &optimizer)
{}
