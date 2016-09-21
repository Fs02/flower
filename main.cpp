#include <iostream>
#include <flower/net.h>
#include <flower/layer/tanh.h>
#include <flower/layer/sigmoid.h>
#include <flower/layer/relu.h>
#include <flower/layer/elu.h>
#include <flower/layer/fully_connected.h>
#include <flower/layer/dropout.h>
#include <flower/optimizer/rms_prop.h>
#include <flower/supervised_learning.h>
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

    flower::EluDef edef;
    flower::ReluDef rdef;
    flower::SigmoidDef sdef;
    flower::TanhDef tdef;
    flower::FullyConnectedDef fdef(3, 3);

    net.add("FullyConnected1", fdef);
    net.add("Elu", edef);
    net.add("FullyConnected2", fdef);
    net.add("Relu", rdef);
    net.add("FullyConnected3", fdef);
    net.add("Sigmoid", sdef);
//    net.add("Dropout", flower::DropoutDef());
    net.add("FullyConnected4", fdef);
    net.add("Tanh", tdef);

    flower::SupervisedLearning trainer(&net, flower::RmsPropDef());

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
