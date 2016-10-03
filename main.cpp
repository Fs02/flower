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
#include <Eigen/CXX11/Tensor>
#include <flower/layer/convolution.h>

using namespace std;

int main()
{
    flower::Net net = flower::Net();

    net.add(flower::FullyConnected(3, 3));
    net.add(flower::Elu());
    net.add(flower::FullyConnected(3, 3));
    net.add(flower::Relu());
    net.add(flower::FullyConnected(3, 3));
    net.add(flower::Sigmoid());

    flower::GradientDescent trainer(&net, flower::Momentum());

    Eigen::Tensor<double, 2> t_data(2, 3);
    t_data.setValues({{0.05, 0.1, -0.5}, {0.1, -0.3, 0.4}});

    Eigen::Tensor<double, 2> t_target(2, 3);
    t_target.setValues({{0.01, 0.99, 1.0}, {0.9, 0.3, 0.2}});

    std::cout << "\n";

    for (int i = 0; i < 50; ++i)
    {
        std::cout << "epoch : "
                  << i
                  << " error: "
                  << trainer.feed(t_data, t_target)
                  << std::endl;
    }


    Eigen::Tensor<double, 4> input(2, 3, 7, 7);
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

    Eigen::Tensor<double, 3> filter(3, 3, 3);
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

    Eigen::Tensor<double, 3> output(2, 3, 3);

    flower::ConvolutionLayer::convolve(input, filter, 1.0, output, 2);

    std::cout << output;

    return 0;
}
