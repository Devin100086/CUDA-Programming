#include <torch/extension.h>
#include "add2.h"
void torch_launch_add2(const torch::Tensor &a_tensor,
                        const torch::Tensor &b_tensor,
                        torch::Tensor &c_tensor,
                        int n) {
    launch_add2((float *)c_tensor.data_ptr(),
                (const float *)a_tensor.data_ptr(),
                (const float *)b_tensor.data_ptr(),
                n);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add2",  // 函数在python中调用的名字
          &torch_launch_add2,  // 函数指针，需绑定的C++函数引用
          "compute element-wise sum of two vectors"  // 函数说明
          );
}