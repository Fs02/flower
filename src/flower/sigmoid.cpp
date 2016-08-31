#include <flower/sigmoid.h>
#include <flower/blob.h>
#include <iostream>

using namespace flower;

inline double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }

Sigmoid::Sigmoid()
{}

inline const char* Sigmoid::type() const
{
    return "Sigmoid";
}

void Sigmoid::forward(Blob& bottom, Blob& top)
{
    auto data = bottom.data().unaryExpr(&sigmoid);
    top.set_data(data);

    //  m_input = input;

    //    m_output->value = sig(m_input->value);
    //m_output->gradient = 0.0;
}

void Sigmoid::backward(Blob& top, Blob& bottom)
{
//    double s = sig(m_input->value);
//    m_input->gradient += (s * (1.0 - s)) * m_output->gradient;
}
