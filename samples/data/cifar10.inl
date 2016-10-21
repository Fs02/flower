template<typename Scalar>
void Cifar10<Scalar>::read_batch(const char *path, Tensor<Scalar, 4> &images, Tensor<Scalar, 2> &labels)
{
    std::ifstream file (path, std::ios::binary);
    if (file.is_open())
    {
        const int n_images = 10000;
        const int n_rows = 32;
        const int n_cols = 32;

        Tensor<Scalar, 4, RowMajor> row_images = Tensor<Scalar, 4, RowMajor>(n_images, 3, n_rows, n_cols);
        labels = Tensor<Scalar, 2>(n_images, 10);
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
                        row_images(i, ch, r, c) = (Scalar)temp;
                    }
                }
            }
        }

        images = row_images.swap_layout();
    }
}
