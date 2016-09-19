#include <iostream>
#include <flower/net.h>
#include <flower/layer/tanh.h>
#include <flower/layer/sigmoid.h>
#include <flower/layer/fully_connected.h>
#include <Eigen/Core>

using namespace std;

int main()
{
    flower::Net net = flower::Net();

    Eigen::MatrixXd data(1, 2);
    data << 0.05, 0.1;

    Eigen::MatrixXd target(1, 2);
    target << 0.01, 0.99;

    flower::SigmoidDef sdef(2);
    flower::TanhDef tdef(2);
    flower::FullyConnectedDef fdef(2, 2);

    net.add("FullyConnected1", fdef);
    net.add("Sigmoid", sdef);
    net.add("FullyConnected2", fdef);
    net.add("Tanh", tdef);

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
