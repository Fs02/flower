#include <iostream>
#include <flower/net.h>
#include <flower/layer/tanh.h>
#include <flower/layer/sigmoid.h>
#include <flower/layer/relu.h>
#include <flower/layer/elu.h>
#include <flower/layer/fully_connected.h>
#include <flower/layer/dropout.h>
#include <flower/gradient_descent.h>
#include <flower/optimizer/adam.h>
#include <Eigen/Core>
#include <Eigen/CXX11/Tensor>

using namespace std;

int main()
{
    flower::Net net = flower::Net();

    // 2 data x 3 features
    Eigen::MatrixXd data(2, 3);
    data << 0.05, 0.1, -0.5,
            0.1, -0.3, 0.4;

    Eigen::MatrixXd target(2, 3);
    target << 0.01, 0.99, 1.0,
              0.9, 0.3, 0.2;

    net.add("FullyConnected1", flower::FullyConnectedDef(3, 3));
    net.add("Elu", flower::EluDef());
    net.add("FullyConnected2", flower::FullyConnectedDef(3, 3));
    net.add("Relu", flower::ReluDef());
    net.add("FullyConnected3", flower::FullyConnectedDef(3, 3));
    net.add("Sigmoid", flower::SigmoidDef());

    flower::GradientDescent trainer(&net, flower::AdamDef());

    for (int i = 0; i < 5; ++i)
    {
        std::cout << "epoch : "
                  << i
                  << " error: "
                  << trainer.feed(data, target)
                  << std::endl;
    }

    Eigen::Tensor<double, 2> t_data(2, 3);
    t_data.setValues({{0.05, 0.1, -0.5}, {0.1, -0.3, 0.4}});

    Eigen::Tensor<double, 2> t_target(2, 3);
    t_target.setValues({{0.01, 0.99, 1.0}, {0.9, 0.3, 0.2}});

    std::cout << "\n";

    for (int i = 0; i < 5; ++i)
    {
        std::cout << "epoch : "
                  << i
                  << " error: "
                  << trainer.feed(t_data, t_target)
                  << std::endl;
    }

    return 0;
}
