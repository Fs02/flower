#include <iostream>
#include <array>
#include <flower/feature.h>
#include <flower/sigmoid.h>
#include <flower/fully_connected.h>
#include <flower/hinge_loss.h>
#include <flower/softmax_loss.h>
#include <Eigen/Core>

using namespace std;

int main()
{
    Eigen::MatrixXd data(4, 1);
    data << 10, 1, 2, 4;

    flower::Feature input = flower::Feature(data);
    flower::Feature full = flower::Feature(data);
    flower::Feature sig = flower::Feature(data);
    flower::Feature loss = flower::Feature(data);

    flower::SigmoidDef sdef("s1", 3);
    flower::Sigmoid s(&sdef);

    flower::FullyConnectedDef fdef("f1", 4, 4);
    flower::FullyConnected f(&fdef);

    flower::HingeLossDef ldef("loss", 1.0);
    flower::HingeLoss l(&ldef);

//    f.forward(input, full);
    s.forward(input, sig);
    l.forward(sig, loss);

    cout << "Forward"
         << "\n Input: \n"
         << input.data()
         << "\n Full: \n"
         << full.data()
         << "\n Sig: \n"
         << sig.data()
         << endl;

    Eigen::MatrixXd score(3, 1);
    score << -2.85, 0.86, 0.28;

    flower::SoftmaxLossDef sldef("sl");
    flower::SoftmaxLoss sl(&sldef);


    flower::Feature sf = flower::Feature(score);
    flower::Feature of = flower::Feature(score);

    sl.forward(sf, of);

    return 0;
}
