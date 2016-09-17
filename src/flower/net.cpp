#include <flower/net.h>

using namespace flower;

#include <iostream>

Net::Net()
    : layers_()
{}

Net::~Net()
{
    layers_.clear();
}

void Net::train(const Eigen::MatrixXd& data, const Eigen::MatrixXd& label)
{
    std::cout << "\ntrain--\n";

    auto feat = data;
    // forward propagate
    for(const auto &layer : layers_)
    {
        std::cout << "\n" << layer.first;
        feat = layer.second->forward(feat);
    }

    std::cout << "\nfeat--\n"
              << feat
              << "\nfeat--";

    auto loss = label - feat;

    for(auto i = layers_.rbegin(); i != layers_.rend(); ++i)
    {
        std::cout << "\n" << (*i).first;
        (*i).second->backward(loss);
    }
}

void Net::eval()
{}

void Net::add(const char *name, const ILayerDef &definition)
{
    layers_.push_back(std::make_pair(name, definition.create(this, name)));
}
