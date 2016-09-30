#include <iostream>
#include <flower/net.h>
#include <flower/layer/tanh.h>
#include <flower/layer/sigmoid.h>
#include <flower/layer/relu.h>
#include <flower/layer/elu.h>
#include <flower/layer/fully_connected.h>
#include <flower/layer/dropout.h>
#include <flower/optimizer/vanilla.h>
#include <flower/gradient_descent.h>
#include <Eigen/Core>

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

    flower::GradientDescent trainer(&net, 0.01f);

    for (int i = 0; i < 100; ++i)
    {
        std::cout << "epoch : "
                  << i
                  << " error: "
                  << trainer.feed(data, target)
                  << std::endl;
    }

    return 0;
}
