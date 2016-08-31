#include <iostream>
#include <array>
#include <flower/blob.h>
#include <flower/sigmoid.h>

using namespace std;

int main()
{
    Eigen::VectorXd data(4);
    data << 0, 1, 2, 3;

    flower::Blob input = flower::Blob(data);
    flower::Blob output = flower::Blob(data);

    flower::Sigmoid s;
    s.forward(input, output);
    cout << s.type()
         << endl
         << input.data()
         << endl
         << output.data();

    return 0;
}

