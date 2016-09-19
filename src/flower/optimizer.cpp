#include <flower/optimizer.h>
#include <flower/net.h>

using namespace flower;

IOptimizerDef::IOptimizerDef()
{}

IOptimizer::IOptimizer(Net* net, const IOptimizerDef &definition)
    : net_(net)
{}
