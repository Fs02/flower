#include <iostream>
#include <array>
#include <flower/net.h>
#include <flower/feature.h>
#include <flower/layer/sigmoid.h>
#include <flower/layer/fully_connected.h>
#include <flower/hinge_loss.h>
#include <flower/softmax_loss.h>
#include <Eigen/Core>

using namespace std;

int main()
{
    flower::Net net = flower::Net();

    Eigen::MatrixXd data(1, 2);
    data << 0.05, 0.1;

    Eigen::MatrixXd label(1, 2);
    label << 0.01, 0.99;

    flower::Feature input = flower::Feature(data);
    flower::Feature full = flower::Feature(data);
    flower::Feature sig = flower::Feature(data);
    flower::Feature loss = flower::Feature(data);

    flower::SigmoidDef sdef(2);
    flower::Sigmoid s(&net, "", sdef);

    flower::FullyConnectedDef fdef(2, 2);
//    flower::FullyConnected f(&net, "f1", fdef);

//    flower::HingeLossDef ldef(1.0);
//    flower::HingeLoss l(&net, "loss", &ldef);

    net.add("FullyConnected1", fdef);
    net.add("Sigmoid1", sdef);
    net.add("FullyConnected2", fdef);
    net.add("Sigmoid2", sdef);
//    net.add("FullyConnected3", fdef);

    for (int i = 0; i < 2; ++i)
    {
        std::cout << "epoch : " << i << endl;
        net.train(data, label);
    }

//    f.forward(input, full);
//    s.forward(input, sig);
//    l.forward(sig, loss);

//    cout << "Forward"
//         << "\n Input: \n"
//         << input.data()
//         << "\n Full: \n"
//         << full.data()
//         << "\n Sig: \n"
//         << sig.data()
//         << endl;

    Eigen::MatrixXd score(3, 1);
    score << -2.85, 0.86, 0.28;

//    flower::SoftmaxLossDef sldef = flower::SoftmaxLossDef();
//    flower::SoftmaxLoss sl(&net, "sl", &sldef);


    flower::Feature sf = flower::Feature(score);
    flower::Feature of = flower::Feature(score);

//    sl.forward(sf, of);

    return 0;
}
