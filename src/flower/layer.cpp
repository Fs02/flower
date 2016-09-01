#include <flower/layer.h>

using namespace flower;

ILayerDef::ILayerDef(const char *name)
    : name_(name)
{}

ILayer::ILayer(ILayerDef *definition)
{}
