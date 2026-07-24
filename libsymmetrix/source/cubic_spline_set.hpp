#pragma once

#include <span>
#include <vector>

class CubicSplineSet {

public:

struct EvaluationPoint {
    int interval;
    double x;
    double xx;
    double xxx;
};

CubicSplineSet(double h,
               std::vector<std::vector<double>> nodal_values,
               std::vector<std::vector<double>> nodal_derivs,
               double x0 = 0.0);

void evaluate(double r, std::span<double> values);
void evaluate_derivs(double r, std::span<double> values, std::span<double> derivs);
EvaluationPoint evaluation_point(double r) const;
double evaluate_function(const EvaluationPoint& point, int function) const;
void evaluate_function_derivs(
    const EvaluationPoint& point,
    int function,
    double& value,
    double& derivative) const;

// TODO: protect this with accessor
int num_splines;

private:

double h;
double x0;
int num_nodes;
std::vector<double> c;

};
