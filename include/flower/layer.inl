template<typename Scalar>
ILayer<Scalar>::ILayerDef()
{}

template<typename Scalar>
ILayerOp<Scalar>::ILayerOp(Net<Scalar>* net, const ILayer<Scalar> &definition)
    : net_(net), definition_(definition)
{}

template<typename Scalar>
const char *ILayerOp<Scalar>::type() const
{
    return definition_.type();
}

template<typename Scalar>
inline const ILayer<Scalar>& ILayerOp<Scalar>::definition()
{
    return definition_;
}

template<typename Scalar>
void ILayerOp<Scalar>::configure(const IOptimizer<Scalar> &optimizer)
{}
