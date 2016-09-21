#include <iostream>
#include <flower/net.h>
#include <flower/layer/tanh.h>
#include <flower/layer/sigmoid.h>
#include <flower/layer/relu.h>
#include <flower/layer/elu.h>
#include <flower/layer/fully_connected.h>
#include <flower/layer/dropout.h>
#include <flower/optimizer/rms_prop.h>
#include <Eigen/Core>

using namespace std;

int main()
{
    flower::Net net = flower::Net();

    Eigen::MatrixXd data(1, 3);
    data << 0.05, 0.1, -0.5;

    Eigen::MatrixXd target(1, 3);
    target << 0.01, 0.99, 1.0;

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
    net.add("Dropout", flower::DropoutDef());
    net.add("FullyConnected4", fdef);
    net.add("Tanh", tdef);

    net.configure(flower::RmsPropDef());

    for (int i = 0; i < 1000; ++i)
    {
        std::cout << "epoch : "
                  << i
                  << " error: "
                  << net.train(data, target)
                  << std::endl;
    }

    return 0;
}
