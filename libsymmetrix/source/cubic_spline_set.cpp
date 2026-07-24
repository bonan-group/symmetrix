#include <vector>
#include <cmath>
#include <stdexcept>

#include "cubic_spline_set.hpp"

CubicSplineSet::CubicSplineSet(
    double h,
    std::vector<std::vector<double>> nodal_values,
    std::vector<std::vector<double>> nodal_derivs,
    double x0)
{
    // TODO: sanitize input
    this->h = h;
    this->x0 = x0;
    num_nodes = nodal_values[0].size();
    num_splines = nodal_values.size();
    c = std::vector<double>(4*num_splines*(num_nodes-1), 0.0);
    for (int i=0; i<num_nodes-1; ++i) {
        for (int j=0; j<num_splines; ++j) {
            c[(4*i)*num_splines+j] = nodal_values[j][i];
            c[(4*i+1)*num_splines+j] = nodal_derivs[j][i];
            c[(4*i+2)*num_splines+j] = (-3*nodal_values[j][i] -2*h*nodal_derivs[j][i]
                                        + 3*nodal_values[j][i+1] - h*nodal_derivs[j][i+1]) / (h*h);
            c[(4*i+3)*num_splines+j] = (2*nodal_values[j][i] + h*nodal_derivs[j][i]
                                        - 2*nodal_values[j][i+1] + h*nodal_derivs[j][i+1]) / (h*h*h);
        }
    }
}

CubicSplineSet::EvaluationPoint CubicSplineSet::evaluation_point(double r) const
{
    int interval = static_cast<int>(std::floor((r-x0)/h));
    double x = r-x0-h*interval;
    if (interval < 0) {
        interval = 0;
        x = 0.0;
    } else if (interval >= num_nodes-1) {
        interval = num_nodes-2;
        x = h;
    }
    const double xx = x*x;
    return {interval, x, xx, xx*x};
}

double CubicSplineSet::evaluate_function(
    const EvaluationPoint& point,
    int function) const
{
    if (function < 0 || function >= num_splines)
        throw std::out_of_range("Cubic spline function index is out of range.");
    const double* coefficients =
        c.data()+4*point.interval*num_splines+function;
    return coefficients[0*num_splines]
        + coefficients[1*num_splines]*point.x
        + coefficients[2*num_splines]*point.xx
        + coefficients[3*num_splines]*point.xxx;
}

void CubicSplineSet::evaluate_function_derivs(
    const EvaluationPoint& point,
    int function,
    double& value,
    double& derivative) const
{
    if (function < 0 || function >= num_splines)
        throw std::out_of_range("Cubic spline function index is out of range.");
    const double* coefficients =
        c.data()+4*point.interval*num_splines+function;
    value = coefficients[0*num_splines]
        + coefficients[1*num_splines]*point.x
        + coefficients[2*num_splines]*point.xx
        + coefficients[3*num_splines]*point.xxx;
    derivative = coefficients[1*num_splines]
        + 2.0*coefficients[2*num_splines]*point.x
        + 3.0*coefficients[3*num_splines]*point.xx;
}

void CubicSplineSet::evaluate(
    double r,
    std::span<double> values)
{
    int i = static_cast<int>(std::floor((r-x0)/h));
    double x = r-x0-h*i;
    if (i < 0) {
        i = 0;
        x = 0.0;
    } else if (i >= num_nodes-1) {
        i = num_nodes-2;
        x = h;
    }
    const double xx = x*x;
    const double xxx = xx*x;
    double* c_i = c.data() + 4*i*num_splines;
    for (int j=0; j<num_splines; ++j)
        values[j] = c_i[j];
    c_i += num_splines;
    for (int j=0; j<num_splines; ++j)
        values[j] += c_i[j]*x;
    c_i += num_splines;
    for (int j=0; j<num_splines; ++j)
        values[j] += c_i[j]*xx;
    c_i += num_splines;
    for (int j=0; j<num_splines; ++j)
        values[j] += c_i[j]*xxx;
}

void CubicSplineSet::evaluate_derivs(double r,
                                     std::span<double> values,
                                     std::span<double> derivs)
{
    int i = static_cast<int>(std::floor((r-x0)/h));
    double x = r-x0-h*i;
    if (i < 0) {
        i = 0;
        x = 0.0;
    } else if (i >= num_nodes-1) {
        i = num_nodes-2;
        x = h;
    }
    const double xx = x*x;
    const double xxx = xx*x;
    const double two_x = 2*x;
    const double three_xx = 3*xx;
    // compute values
    double* c_i = c.data() + 4*i*num_splines;
    for (int j=0; j<num_splines; ++j)
        values[j] = c_i[j];
    c_i += num_splines;
    for (int j=0; j<num_splines; ++j)
        values[j] += x*c_i[j];
    c_i += num_splines;
    for (int j=0; j<num_splines; ++j)
        values[j] += xx*c_i[j];
    c_i += num_splines;
    for (int j=0; j<num_splines; ++j)
        values[j] += xxx*c_i[j];
    // compute derivs
    c_i = c.data() + (4*i+1)*num_splines;
    for (int j=0; j<num_splines; ++j)
        derivs[j] = c_i[j];
    c_i += num_splines;
    for (int j=0; j<num_splines; ++j)
        derivs[j] += two_x*c_i[j];
    c_i += num_splines;
    for (int j=0; j<num_splines; ++j)
        derivs[j] += three_xx*c_i[j];
}
