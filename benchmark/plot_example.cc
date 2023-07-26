// example from https://matplotlib.org/stable/gallery/scales/aspect_loglog.html

#include <vector>

#if defined(TACHYON_HAS_MATPLOTLIB)
#include "third_party/matplotlibcpp17/include/animation.h"
#include "third_party/matplotlibcpp17/include/pyplot.h"

using namespace std;
using namespace matplotlibcpp17;

int main() {
  py::scoped_interpreter guard{};
  auto plt = pyplot::import();
  auto [fig, axs] = plt.subplots(1, 2);
  auto &ax1 = axs[0], ax2 = axs[1];
  ax1.set_xscale(Args("log"));
  ax1.set_yscale(Args("log"));
  ax1.set_xlim(Args(1e+1, 1e+3));
  ax1.set_ylim(Args(1e+2, 1e+3));
  ax1.set_aspect(Args(1));
  ax1.set_title(Args("adjustable = box"));

  ax2.set_xscale(Args("log"));
  ax2.set_yscale(Args("log"));
  ax2.set_adjustable(Args("datalim"));
  ax2.plot(Args(vector<int>({1, 3, 10}), vector<int>({1, 9, 100}), "o-"));
  ax2.set_xlim(Args(1e-1, 1e+2));
  ax2.set_ylim(Args(1e-1, 1e+3));
  ax2.set_aspect(Args(1));
  ax2.set_title(Args("adjustable = datalim"));

  plt.show();
}
#else
#include <iostream>

int main() {
  std::cerr << "Please build with --//:tachyon_has_matplotlib" << std::endl;
  return 0;
}
#endif
