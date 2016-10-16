#include <iostream>
#include <flower/net.h>
#include <flower/layer/tanh.h>
#include <flower/layer/sigmoid.h>
#include <flower/layer/relu.h>
#include <flower/layer/elu.h>
#include <flower/layer/fully_connected.h>
#include <flower/layer/dropout.h>
#include <flower/gradient_descent.h>
#include <flower/optimizer/momentum.h>
#include <flower/tensor.h>
#include <flower/layer/convolution.h>
#include <flower/layer/pooling.h>

using namespace std;

int main()
{
    const int order = 3;
    flower::Tensor<double, order> test;

    flower::Net net = flower::Net(10);

    net.add(flower::FullyConnected(3, 3));
    net.add(flower::Elu());
    net.add(flower::FullyConnected(3, 3));
    net.add(flower::Relu());
    net.add(flower::FullyConnected(3, 3));
    net.add(flower::Sigmoid());

    flower::GradientDescent trainer(&net, flower::Momentum());

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

    flower::TensorData<double> testdata(10);
    double *storage = new double[10];  // 2 x 4 x 2 x 8 = 128
    double *p = testdata.data().get();
    Eigen::TensorMap<Eigen::Tensor<double, 2>> t(storage, 2, 5);
    delete [] storage;
    std::cout << t;
    std::cout << std::endl << testdata.map<2>(5, 2);

    return 0;
}
