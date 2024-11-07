#include <cmath>
#include <functional>

struct ActivationFunction {
  std::function<double(double)> function;

  // Default constructor sets sigmoid as the default function
  ActivationFunction()
      : function([](double x) { return 1.0 / (1.0 + std::exp(-x)); })
  {
  }
  // Overload operator() to call the function directly
  const double operator()(double input) const { return function(input); }
};

// Sigmoid function implementation
inline double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
