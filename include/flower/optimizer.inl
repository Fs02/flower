template<typename Scalar>
IOptimizer<Scalar>::IOptimizer()
{}

template<typename Scalar>
IOptimizerOp<Scalar>::IOptimizerOp(Net<Scalar>* net, const IOptimizer<Scalar> &definition)
    : net_(net)
{}
