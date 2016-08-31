#include <flower/sigmoid.h>
#include <flower/blob.h>
#include <iostream>

using namespace flower;

inline double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }

Sigmoid::Sigmoid()
{}

void Sigmoid::forward(Blob& bottom, Blob& top)
{
    auto data = bottom.data().unaryExpr(&sigmoid);
    top.set_data(data);
}

void Sigmoid::backward(Blob& top, Blob& bottom)
{
    // calculate derivative in one pass
    // d = (sigmoid(bottom.data) * (1.0 - sigmoid(bottom.data))) * top.diff
    auto d = bottom.data().unaryExpr([](double x) { return sigmoid(x) * (1.0 - sigmoid(x)); }).cwiseProduct(top.diff());
    bottom.set_diff(bottom.diff() + d);
}
