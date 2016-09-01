#include <iostream>
#include <array>
#include <flower/feature.h>
#include <flower/sigmoid.h>
#include <flower/fully_connected.h>

using namespace std;

int main()
{
    Eigen::MatrixXd data(4, 1);
    data << 0, 1, 2, 3;

    flower::Feature input = flower::Feature(data);
    flower::Feature full = flower::Feature(data);
    flower::Feature sig = flower::Feature(data);

    flower::SigmoidDef sdef("s1", 3);
    flower::Sigmoid s(&sdef);

    flower::FullyConnectedDef fdef("f1", 4, 4);
    flower::FullyConnected f(&fdef);

    f.forward(input, full);
    s.forward(full, sig);

    cout << "Forward"
         << "\n Input: \n"
         << input.data()
         << "\n Full: \n"
         << full.data()
         << "\n Sig: \n"
         << sig.data()
         << endl;

    return 0;
}
