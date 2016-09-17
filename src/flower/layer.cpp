#include <flower/layer.h>

using namespace flower;

ILayerDef::ILayerDef()
{}

ILayer::ILayer(Net* net, const char *name, const ILayerDef &definition, unsigned int size_x, unsigned size_y)
    : net_(net), name_(name), feat_(size_x, size_y), diff_(size_x, size_y)
{}
