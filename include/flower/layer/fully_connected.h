#ifndef FLOWER_FULLY_CONNECTED_H
#define FLOWER_FULLY_CONNECTED_H

#include <flower/layer.h>

namespace flower
{
    class FullyConnectedDef : public ILayerDef
    {
    public:
        FullyConnectedDef(unsigned int bottom_size, unsigned int top_size);

        inline const char *type() const { return "FullyConnected"; }

        inline unsigned int bottom_size() const { return bottom_size_; }
        inline unsigned int top_size() const { return top_size_; }

    protected:
        layer_ptr create(Net *net, const char* name) const;

        unsigned int bottom_size_;
        unsigned int top_size_;
    };

    class FullyConnected : public ILayer
    {
    public:
        FullyConnected(Net* net, const char *name, const FullyConnectedDef &definition);

        inline const char *type() const { return "FullyConnected"; }

        Eigen::MatrixXd forward(const Eigen::MatrixXd &data);
        Eigen::MatrixXd backward(const Eigen::MatrixXd &errors);

        static int count;

    protected:
        Eigen::MatrixXd data_;
        Eigen::MatrixXd weights_;
        Eigen::MatrixXd bias_;
    };
}

#endif
