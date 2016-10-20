#include <iostream>
#include <flower/tensor.h>
#include <flower/net.h>
#include <flower/layer/tanh.h>
#include <flower/layer/sigmoid.h>
#include <flower/layer/relu.h>
#include <flower/layer/elu.h>
#include <flower/layer/fully_connected.h>
#include <flower/gradient_descent.h>
#include <flower/optimizer/vanilla.h>
using namespace std;

int main()
{
    auto net = flower::Net<double>();

    net.add_layer(flower::FullyConnected<double>(3, 3));
    net.add_layer(flower::Elu<double>());
    net.add_layer(flower::FullyConnected<double>(3, 3));
    net.add_layer(flower::Relu<double>());
    net.add_layer(flower::FullyConnected<double>(3, 3));
    net.add_layer(flower::Sigmoid<double>());

    flower::GradientDescent<double> trainer(&net, flower::Vanilla<double>());

    flower::Tensor<double, 2> t_data(2, 3);
    t_data.setValues({{0.05, 0.1, -0.5}, {0.1, -0.3, 0.4}});

    flower::Tensor<double, 2> t_target(2, 3);
    t_target.setValues({{0.01, 0.99, 1.0}, {0.9, 0.3, 0.2}});

    std::cout << "\n";

    for (int i = 0; i < 1; ++i)
    {
        std::cout << "epoch : "
                  << i
                  << " error: "
                  << trainer.feed(t_data, t_target)
                  << std::endl;
    }

    /*
    flower::Tensor<double, 4> input(2, 3, 7, 7);
    input.setConstant(1.0);
    input.setValues({
                        {
                            {
                                {0, 0, 0, 0, 0, 0, 0},
                                {0, 0, 2, 0, 1, 1, 0},
                                {0, 0, 0, 0, 1, 1, 0},
                                {0, 0, 1, 2, 2, 1, 0},
                                {0, 2, 1, 1, 1, 2, 0},
                                {0, 0, 1, 1, 1, 2, 0},
                                {0, 0, 0, 0, 0, 0, 0}
                            },
                            {
                                {0, 0, 0, 0, 0, 0, 0},
                                {0, 1, 1, 0, 1, 2, 0},
                                {0, 1, 0, 1, 2, 2, 0},
                                {0, 2, 0, 1, 2, 0, 0},
                                {0, 1, 0, 2, 1, 1, 0},
                                {0, 0, 2, 1, 1, 2, 0},
                                {0, 0, 0, 0, 0, 0, 0}
                            },
                            {
                                {0, 0, 0, 0, 0, 0, 0},
                                {0, 0, 2, 1, 0, 2, 0},
                                {0, 2, 0, 0, 2, 2, 0},
                                {0, 1, 1, 2, 2, 1, 0},
                                {0, 2, 1, 2, 2, 1, 0},
                                {0, 0, 2, 2, 2, 0, 0},
                                {0, 0, 0, 0, 0, 0, 0}
                            }
                        },
                        {
                            {
                                {0, 0, 0, 0, 0, 0, 0},
                                {0, 0, 2, 0, 1, 1, 0},
                                {0, 0, 0, 0, 1, 1, 0},
                                {0, 0, 1, 2, 2, 1, 0},
                                {0, 2, 1, 1, 1, 2, 0},
                                {0, 0, 1, 1, 1, 2, 0},
                                {0, 0, 0, 0, 0, 0, 0}
                            },
                            {
                                {0, 0, 0, 0, 0, 0, 0},
                                {0, 1, 1, 0, 1, 2, 0},
                                {0, 1, 0, 1, 2, 2, 0},
                                {0, 2, 0, 1, 2, 0, 0},
                                {0, 1, 0, 2, 1, 1, 0},
                                {0, 0, 2, 1, 1, 2, 0},
                                {0, 0, 0, 0, 0, 0, 0}
                            },
                            {
                                {0, 0, 0, 0, 0, 0, 0},
                                {0, 0, 2, 1, 0, 2, 0},
                                {0, 2, 0, 0, 2, 2, 0},
                                {0, 1, 1, 2, 2, 1, 0},
                                {0, 2, 1, 2, 2, 1, 0},
                                {0, 0, 2, 2, 2, 0, 0},
                                {0, 0, 0, 0, 0, 0, 0}
                            }
                        }
                    });

    flower::Tensor<double, 3> filter(3, 3, 3);
    filter.setValues({
                         {
                             {-1, 0, 0},
                             {-1,-1, 0},
                             { 0, 0, 1}
                         },
                         {
                             { 0, 0,-1},
                             { 1,-1, 1},
                             { 0, 0, 1}
                         },
                         {
                             {-1, 1,-1},
                             { 0,-1, 0},
                             { 1,-1,-1}
                         }
                     });

    flower::Tensor<double, 3> output(2, 3, 3);

    flower::ConvolutionLayer::convolve(input, output, filter, 1.0, 2);

    std::cout << output << std::endl;

    flower::ConvolutionLayer conv(&net, flower::Convolution({3, 3, 3}, 2, 2, 0));
    std::cout << conv.forward(input).size();

    flower::Tensor<double, 4> o(2, 3, 6, 6);
    flower::array<int, 2> size({2, 2});
    flower::PoolingLayer::pool(flower::Pooling::Mode::Max, input, o, size, 1);
    std::cout << o;
    */

    double a[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    flower::TensorData<double> testdata(a, 10);
    double *storage = new double[10];  // 2 x 4 x 2 x 8 = 128
    double *p = testdata.data();
    Eigen::TensorMap<Eigen::Tensor<double, 2>> t(storage, 2, 5);
    delete [] storage;
    std::cout << t;
    std::cout << std::endl << testdata.map<1>(testdata.size());

    flower::Tensor<double, 2> t2 = testdata.map<2>(2, 5);
    flower::array<std::pair<int, int>, 2> bias({std::make_pair(0, 0), std::make_pair(0, 1)});
    flower::Tensor<double, 2> t3 = t2.pad(bias, 1);

    std::cout << std::endl << t3;

    return 0;
}
