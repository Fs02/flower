#include <flower/feature.h>

using namespace flower;

Feature::Feature(unsigned int x, unsigned int y)
    : data_(x, y), diff_(x, y)
{}

Feature::Feature(const Eigen::MatrixXd &data)
    : data_(data), diff_(data)
{}

const Eigen::MatrixXd &Feature::data() const
{
    return data_;
}

const Eigen::MatrixXd &Feature::diff() const
{
    return diff_;
}

void Feature::set_data(const Eigen::MatrixXd &data)
{
    data_ = data;
}

void Feature::set_diff(const Eigen::MatrixXd &diff)
{
    diff_ = diff;
}
