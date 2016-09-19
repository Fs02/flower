#ifndef FLOWER_FULLY_CONNECTED_H
#define FLOWER_FULLY_CONNECTED_H

#include <flower/layer.h>

namespace flower
{
    class FullyConnectedDef : public ILayerDef
    {
    public:
        FullyConnectedDef(unsigned int input_size, unsigned int output_size);

        inline const char *type() const { return "FullyConnected"; }

        inline unsigned int input_size() const { return input_size_; }
        inline unsigned int output_size() const { return output_size_; }

    protected:
        layer_ptr create(Net *net, const char* name) const;

        unsigned int input_size_;
        unsigned int output_size_;
    };

    class FullyConnected : public ILayer
    {
    public:
        FullyConnected(Net* net, const char *name, const FullyConnectedDef &definition);

        inline const char *type() const { return "FullyConnected"; }

        Eigen::MatrixXd forward(const Eigen::MatrixXd &data);
        Eigen::MatrixXd backward(const Eigen::MatrixXd &errors);

    protected:
        Eigen::MatrixXd data_;
        Eigen::MatrixXd weights_;
        Eigen::MatrixXd bias_;
    };
}

#endif
