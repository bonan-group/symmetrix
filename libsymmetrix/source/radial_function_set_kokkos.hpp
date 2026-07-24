#pragma once

#include <vector>

#include <Kokkos_Core.hpp>

template <typename Precision>
class RadialFunctionSetKokkos
{
public:

    struct EvaluationPoint {
        int interval;
        double x;
        double xx;
        double xxx;
    };
    
    RadialFunctionSetKokkos();
    RadialFunctionSetKokkos(
        double h,
        std::vector<std::vector<std::vector<double>>> node_values,
        std::vector<std::vector<std::vector<double>>> node_derivatives,
        double x0 = 0.0);
    void evaluate(
        const int num_nodes,
        Kokkos::View<const int*> node_types,
        Kokkos::View<const int*> num_neigh,
        Kokkos::View<const int*> neigh_types,
        Kokkos::View<const int*> type_to_active,
        int num_active_types,
        Kokkos::View<const double*> r,
        Kokkos::View<Precision**,Kokkos::LayoutRight> R,
        Kokkos::View<Precision**,Kokkos::LayoutRight> R_deriv) const;

    KOKKOS_INLINE_FUNCTION
    EvaluationPoint evaluation_point(double radius) const
    {
        int interval = static_cast<int>(Kokkos::floor((radius-x0)/h));
        double x = radius-x0-h*interval;
        const int num_intervals = num_nodes-1;
        if (interval < 0) {
            interval = 0;
            x = 0.0;
        } else if (interval >= num_intervals) {
            interval = num_intervals-1;
            x = h;
        }
        const double xx = x*x;
        return {interval, x, xx, xx*x};
    }

    KOKKOS_INLINE_FUNCTION
    Precision evaluate_function(
        int edge_type,
        const EvaluationPoint& point,
        int function) const
    {
        const Precision c0 = coefficients(edge_type,point.interval,0,function);
        const Precision c1 = coefficients(edge_type,point.interval,1,function);
        const Precision c2 = coefficients(edge_type,point.interval,2,function);
        const Precision c3 = coefficients(edge_type,point.interval,3,function);
        return c0+c1*static_cast<Precision>(point.x)
            +c2*static_cast<Precision>(point.xx)
            +c3*static_cast<Precision>(point.xxx);
    }

    KOKKOS_INLINE_FUNCTION
    Precision evaluate_function(
        int edge_type,
        double radius,
        int function) const
    {
        return evaluate_function(edge_type, evaluation_point(radius), function);
    }

    KOKKOS_INLINE_FUNCTION
    void evaluate_function(
        int edge_type,
        const EvaluationPoint& point,
        int function,
        Precision& value,
        Precision& derivative) const
    {
        const Precision c0 = coefficients(edge_type,point.interval,0,function);
        const Precision c1 = coefficients(edge_type,point.interval,1,function);
        const Precision c2 = coefficients(edge_type,point.interval,2,function);
        const Precision c3 = coefficients(edge_type,point.interval,3,function);
        value = c0+c1*static_cast<Precision>(point.x)
            +c2*static_cast<Precision>(point.xx)
            +c3*static_cast<Precision>(point.xxx);
        derivative = c1+c2*static_cast<Precision>(2.0*point.x)
            +c3*static_cast<Precision>(3.0*point.xx);
    }

    KOKKOS_INLINE_FUNCTION
    void evaluate_function(
        int edge_type,
        double radius,
        int function,
        Precision& value,
        Precision& derivative) const
    {
        evaluate_function(
            edge_type, evaluation_point(radius), function, value, derivative);
    }
    
private:
    
    double h;
    double x0;
    int num_edge_types;
    int num_functions;
    int num_nodes;
    Kokkos::View<const Precision****,Kokkos::LayoutRight> coefficients;
};
