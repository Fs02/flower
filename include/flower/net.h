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

        template<class T>
        void add(const char *name, ILayerDef* definition);

    private:
        std::unordered_map<const char*, ILayer*> layers_;
    };
}

#include <flower/net.inl>

#endif
