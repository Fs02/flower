#include <flower/feature.h>

using namespace flower;

Feature::Feature(unsigned int size)
    : data_(size), diff_(size)
{}

Feature::Feature(const Eigen::VectorXd &data)
    : data_(data), diff_(data)
{}

const Eigen::VectorXd &Feature::data() const
{
    return data_;
}

const Eigen::VectorXd &Feature::diff() const
{
    return diff_;
}

void Feature::set_data(const Eigen::VectorXd &data)
{
    data_ = data;
}

void Feature::set_diff(const Eigen::VectorXd &diff)
{
    diff_ = diff;
}
