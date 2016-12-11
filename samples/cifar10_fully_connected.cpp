#include <iostream>
#include <flower/tensor.h>
#include <flower/net.h>
#include <flower/layer/sigmoid.h>
#include <flower/layer/fully_connected.h>
#include <flower/gradient_descent.h>
#include <flower/optimizer/vanilla.h>

#include <data/cifar10.h>

using namespace std;

int main()
{
    flower::Cifar10<double> cifar10;
    flower::Tensor<double, 4, flower::RowMajor> images;
    flower::Tensor<double, 2, flower::RowMajor> labels;
    cifar10.read_batch("cifar-10-batches-bin/data_batch_1.bin", images, labels);

    auto net = flower::Net<double>();

    net.add_layer(flower::FullyConnected<double>(3072, 1536));
    net.add_layer(flower::Sigmoid<double>());
    net.add_layer(flower::FullyConnected<double>(1536, 10));
    net.add_layer(flower::Sigmoid<double>());

    flower::GradientDescent<double> trainer(&net, flower::Vanilla<double>());

    for (int i = 0; i < 1; ++i)
    {
        std::cout << "epoch : "
                  << i
                  << " error: "
                  << trainer.feed(images, labels)
                  << std::endl;
    }

    return 0;
}
