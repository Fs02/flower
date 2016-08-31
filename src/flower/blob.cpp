#include <flower/blob.h>

using namespace flower;

Blob::Blob(unsigned int size)
    : data_(size), diff_(size)
{}

Blob::Blob(const Eigen::VectorXd& data)
    : data_(data), diff_(data)
{}

const Eigen::VectorXd& Blob::data() const
{
    return data_;
}

const Eigen::VectorXd& Blob::diff() const
{
    return diff_;
}

void Blob::set_data(const Eigen::VectorXd& data)
{
    data_ = data;
}

void Blob::set_diff(const Eigen::VectorXd& diff)
{
    diff_ = diff;
}

