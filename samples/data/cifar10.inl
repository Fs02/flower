template<typename Scalar>
void Cifar10<Scalar>::read_batch(const char *path, Tensor<Scalar, 4, RowMajor> &images, Tensor<Scalar, 2, RowMajor> &labels)
{
    std::ifstream file (path, std::ios::binary);
    if (file.is_open())
    {
        const int n_images = 2;
        const int n_rows = 32;
        const int n_cols = 32;

        labels = Tensor<Scalar, 2, RowMajor>(n_images, 10);
        labels.setZero();

        for(int i = 0; i < n_images; ++i)
        {
            unsigned char tplabel = 0;
            file.read((char*) &tplabel, sizeof(tplabel));
            labels(i, (int)tplabel) = 1;

            for(int ch = 0; ch < 3; ++ch){
                for(int r = 0; r < n_rows; ++r){
                    for(int c = 0; c < n_cols; ++c){
                        unsigned char temp = 0;
                        file.read((char*) &temp, sizeof(temp));
                        images(i, ch, r, c) = (Scalar)temp;
                    }
                }
            }
        }
    }
}

template<typename Scalar>
void Cifar10<Scalar>::read_batch(const char *path, Tensor<Scalar, 2, RowMajor> &images, Tensor<Scalar, 2, RowMajor> &labels) {
    Tensor<Scalar, 4, RowMajor> images_4d;
    read_batch(path, images_4d, labels);
    Eigen::array<Eigen::DenseIndex, 2> dims({images_4d.dimension(0), 3*32*32});
    images = images_4d.reshape(dims);
}
