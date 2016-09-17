#include <flower/net.h>

using namespace flower;

#include <iostream>

double square(double x) {
    return x * x;
}

Net::Net()
    : layers_()
{}

Net::~Net()
{
    layers_.clear();
}

void Net::train(const Eigen::MatrixXd& data, const Eigen::MatrixXd& label)
{
//    std::cout << "\nforw--\n";

    Eigen::MatrixXd feat = data;
    // forward propagate
    for(const auto &layer : layers_)
    {
        std::cout << "\n" << layer.first;
        feat = layer.second->forward(feat);
        std::cout << "\n" << feat
                  << "\n" << feat.rows() << " x " << feat.cols();
    }

    std::cout << "\nback\n";

    // mse
    std::cout << "ERROR :"
              << (label - feat).unaryExpr(&square).sum() * 0.5
              << std::endl;

    Eigen::MatrixXd loss = -(label - feat).transpose();
    std::cout << "\n"
              << loss
              << "\n" << loss.rows() << " x " << loss.cols();

    for(auto i = layers_.rbegin(); i != layers_.rend(); ++i)
    {
        std::cout << "\n" << (*i).first;
        loss = (*i).second->backward(loss);
        std::cout << "\n" << loss
                  << "\n" << loss.rows() << " x " << loss.cols();
    }

    std::cout << "\nloss\n"
              << loss
              << "\nloss\n";
}

void Net::eval()
{}

void Net::add(const char *name, const ILayerDef &definition)
{
    layers_.push_back(std::make_pair(name, definition.create(this, name)));
}
