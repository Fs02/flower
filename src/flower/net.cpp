#include <flower/net.h>

using namespace flower;

Net::Net()
    : layers_()
{}

Net::~Net()
{
    for(const auto &layer : layers_)
    {
        delete layer.second;
    }
    layers_.clear();
}

void Net::train(const std::vector<Eigen::MatrixXd>& data, const std::vector<Eigen::MatrixXd>& labels)
{}

void Net::eval()
{}
