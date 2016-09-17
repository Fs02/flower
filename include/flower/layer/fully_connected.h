#ifndef FLOWER_FULLY_CONNECTED_H
#define FLOWER_FULLY_CONNECTED_H

#include <flower/layer.h>
#include <flower/net.h>
#include <flower/feature.h>

namespace flower
{
    class FullyConnectedDef : public ILayerDef
    {
    public:
        FullyConnectedDef(unsigned int bottom_size, unsigned int top_size);

        inline const char *type() const { return "FullyConnected"; }

        inline unsigned int bottom_size() const { return bottom_size_; }
        inline unsigned int top_size() const { return top_size_; }

        layer_ptr create(Net *net, const char* name) const;

    protected:
        unsigned int bottom_size_;
        unsigned int top_size_;
    };

    class FullyConnected : public ILayer
    {
    public:
        FullyConnected(Net* net, const char *name, const FullyConnectedDef &definition);

        inline const char *type() const { return "FullyConnected"; }

        void forward(Feature &bottom, Feature &top);
        void backward(Feature &top, Feature &bottom);

        Eigen::MatrixXd forward(const Eigen::MatrixXd &bottom_feat);
        Eigen::MatrixXd backward(const Eigen::MatrixXd &top_diff);

        static int count;

    protected:
        Eigen::MatrixXd weight_feat;
        Eigen::MatrixXd weight_diff;
        Eigen::MatrixXd bias_feat;
        Eigen::MatrixXd bias_diff;

        Feature weights_;
        Feature bias_;
    };
}

#endif
