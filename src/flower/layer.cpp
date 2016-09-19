#include <flower/layer.h>
#include <flower/net.h>

using namespace flower;

ILayerDef::ILayerDef()
{}

ILayer::ILayer(Net* net, const char *name, const ILayerDef &definition)
    : net_(net), name_(name)
{}
