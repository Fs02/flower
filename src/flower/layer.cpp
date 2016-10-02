#include <flower/layer.h>
#include <flower/net.h>

using namespace flower;

ILayerDef::ILayerDef()
{}

ILayer::ILayer(Net* net, const ILayerDef &definition)
    : net_(net)
{}

void ILayer::configure(const IOptimizerDef &optimizer_def)
{}
