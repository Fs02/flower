#include <flower/layer.h>

using namespace flower;

ILayerDef::ILayerDef()
{}

ILayer::ILayer(Net* net, const char *name, ILayerDef *definition)
    : net_(net), name_(name)
{}
