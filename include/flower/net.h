#ifndef FLOWER_NET_H
#define FLOWER_NET_H

#include <flower/layer.h>
#include <Eigen/Core>
#include <vector>
#include <unordered_map>

namespace flower {
    class Net
    {
    public:
        Net();
        ~Net();

        void train(const std::vector<Eigen::MatrixXd>& data, const std::vector<Eigen::MatrixXd>& labels);
        void eval();

    private:
        std::unordered_map<char*, ILayer*> layers_;
    };
}

#endif
