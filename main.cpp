#include <iostream>
#include <array>
#include <flower/feature.h>
#include <flower/sigmoid.h>

using namespace std;

int main()
{
    Eigen::MatrixXd data(4, 1);
    data << 0, 1, 2, 3;

    flower::Feature input = flower::Feature(data);
    flower::Feature output = flower::Feature(data);

    flower::Sigmoid s;
    s.forward(input, output);
    s.backward(output, input);
    cout << "Forward"
         << endl
         << input.data()
         << endl
         << output.data()
         << endl
         << "Backward"
         << endl
         << input.diff()
         << endl
         << output.diff();

    return 0;
}
