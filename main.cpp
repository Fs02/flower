/*
 * Sanity check based on tutorial: https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
 */

#include <iostream>
#include <flower/tensor.h>
#include <flower/net.h>
#include <flower/layer/sigmoid.h>
#include <flower/layer/fully_connected.h>
#include <flower/gradient_descent.h>
#include <flower/optimizer/vanilla.h>
using namespace std;

int main()
{
    auto net = flower::Net<float>();

    net.add_layer(flower::FullyConnected<float>(2, 2));
    net.add_layer(flower::Sigmoid<float>());
    net.add_layer(flower::FullyConnected<float>(2, 2));
    net.add_layer(flower::Sigmoid<float>());

    flower::GradientDescent<float> trainer(&net, flower::Vanilla<float>(0.5));

    // input
    flower::Tensor<float, 2, flower::RowMajor> t_data(1.f, 2.f);
    t_data.setValues({{0.05f, 0.1f}});

    // output
    flower::Tensor<float, 2, flower::RowMajor> t_target(1, 2);
    t_target.setValues({{0.01f, 0.99f}});

    auto fc1 = std::dynamic_pointer_cast<flower::FullyConnectedOp<float>>(net.layers()[0]);
    auto fc2 = std::dynamic_pointer_cast<flower::FullyConnectedOp<float>>(net.layers()[2]);

    // initial weights
    fc1->weights().setValues({{0.15f, 0.25f}, {0.20f, 0.30f}, {0.35f, 0.35f}});
    fc2->weights().setValues({{0.40f, 0.50f}, {0.45f, 0.55f}, {0.60f, 0.60f}});

    std::cout << "\n==initial weight=="
              << "\nW1:\n"
              << fc1->weights()
              << "\nW2:\n"
              << fc2->weights();

    std::cout << "\n";

    for (int i = 0; i < 1; ++i)
    {
        std::cout << "\nepoch : "
                  << i
                  << " error: "
                  << trainer.feed(t_data, t_target);
    }

    std::cout << "\n==updated weight=="
              << "\nW1:\n"
              << fc1->weights()
              << "\nW2:\n"
              << fc2->weights();
    return 0;
}
