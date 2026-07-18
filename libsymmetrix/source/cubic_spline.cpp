#include <cmath>
#include <stdexcept>
#include <vector>

#include "cubic_spline.hpp"

CubicSpline::CubicSpline(
    double h,
    std::vector<double> nodal_values,
    std::vector<double> nodal_derivs,
    double x0)
    : h(h),
      x0(x0),
      c(generate_coefficients(h, nodal_values, nodal_derivs))
{
}

double CubicSpline::evaluate(double r)
{
    const int num_intervals = c.size()/4;
    int i = static_cast<int>(std::floor((r-x0)/h));
    double x = r-x0-h*i;
    if ((i < 0 || i >= num_intervals) && x0 == 0.0)
        throw std::invalid_argument("Out of bounds in CubicSpline::evaluate.");
    if (i < 0) {
        i = 0;
        x = 0.0;
    } else if (i >= num_intervals) {
        i = num_intervals-1;
        x = h;
    }
    const double xx = x*x;
    const double xxx = xx*x;
    const int i4 = 4*i;
    const double c0 = c[i4], c1 = c[i4+1], c2=c[i4+2], c3=c[i4+3];
    return c0 + c1*x + c2*xx + c3*xxx;
}

std::tuple<double,double> CubicSpline::evaluate_deriv(double r)
{
    const int num_intervals = c.size()/4;
    int i = static_cast<int>(std::floor((r-x0)/h));
    double x = r-x0-h*i;
    if ((i < 0 || i >= num_intervals) && x0 == 0.0)
        throw std::invalid_argument("Out of bounds in CubicSpline::evaluate_deriv.");
    if (i < 0) {
        i = 0;
        x = 0.0;
    } else if (i >= num_intervals) {
        i = num_intervals-1;
        x = h;
    }
    const double xx = x*x;
    const double xxx = xx*x;
    const int i4 = 4*i;
    const double c0 = c[i4], c1 = c[i4+1], c2=c[i4+2], c3=c[i4+3];
    return {c0 + c1*x + c2*xx + c3*xxx, c1 + 2*c2*x + 3*c3*xx};
}

std::tuple<double,double> CubicSpline::evaluate_deriv_divided(double r)
{
    const int num_intervals = c.size()/4;
    int i = static_cast<int>(std::floor((r-x0)/h));
    double x = r-x0-h*i;
    if ((i < 0 || i >= num_intervals) && x0 == 0.0)
        throw std::invalid_argument("Out of bounds in CubicSpline::evaluate_deriv.");
    if (i < 0) {
        i = 0;
        x = 0.0;
    } else if (i >= num_intervals) {
        i = num_intervals-1;
        x = h;
    }
    const double xx = x*x;
    const double xxx = xx*x;
    const int i4 = 4*i;
    const double c0 = c[i4], c1 = c[i4+1], c2=c[i4+2], c3=c[i4+3];
    return {c0 + c1*x + c2*xx + c3*xxx, (c1 + 2*c2*x + 3*c3*xx) / r};
}

auto CubicSpline::generate_coefficients(
    double h,
    std::vector<double> nodal_values,
    std::vector<double> nodal_derivs)
    -> std::vector<double>
{
    auto c = std::vector<double>(4*(nodal_values.size()-1), 0.0);
    for (int i=0; i<nodal_values.size()-1; ++i) {
        c[4*i] = nodal_values[i];
        c[4*i+1] = nodal_derivs[i];
        c[4*i+2] = (-3*c[4*i] -2*h*c[4*i+1] + 3*nodal_values[i+1] - h*nodal_derivs[i+1]) / (h*h);
        c[4*i+3] = (2*c[4*i] + h*c[4*i+1] - 2*nodal_values[i+1] + h*nodal_derivs[i+1]) / (h*h*h);
    }
    return c;
}
