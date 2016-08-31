#ifndef BLOB_H
#define BLOB_H

#include <Eigen/Core>

namespace flower {
    class Blob
    {
    public:
        Blob(unsigned int size);
        Blob(const Eigen::VectorXd& data);

        const Eigen::VectorXd& data() const;
        const Eigen::VectorXd& diff() const;

        void set_data(const Eigen::VectorXd& data);
        void set_diff(const Eigen::VectorXd& diff);

    private:
        Eigen::VectorXd data_;
        Eigen::VectorXd diff_;
    };
}

#endif
