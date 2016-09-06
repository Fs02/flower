#ifndef FLOWER_BLOB_H
#define FLOWER_BLOB_H

#include <Eigen/Core>

namespace flower {
    class Feature
    {
    public:
        Feature(unsigned int x, unsigned int y);
        Feature(const Eigen::MatrixXd &data);

        const Eigen::MatrixXd &data() const;
        const Eigen::MatrixXd &diff() const;

        void set_data(const Eigen::MatrixXd &data);
        void set_diff(const Eigen::MatrixXd &diff);

    private:
        Eigen::MatrixXd data_;
        Eigen::MatrixXd diff_;
    };
}

#endif
