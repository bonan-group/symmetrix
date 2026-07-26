#pragma once

#include <type_traits>

#include "../external/cblas-prototypes/cblas.h"

template <typename Precision>
inline void symmetrix_blas_gemm(
    const CBLAS_ORDER order,
    const CBLAS_TRANSPOSE trans_a,
    const CBLAS_TRANSPOSE trans_b,
    const int m,
    const int n,
    const int k,
    const Precision alpha,
    const Precision* a,
    const int lda,
    const Precision* b,
    const int ldb,
    const Precision beta,
    Precision* c,
    const int ldc)
{
    static_assert(std::is_same_v<Precision, float>
                  || std::is_same_v<Precision, double>);
    if constexpr (std::is_same_v<Precision, float>)
        cblas_sgemm(
            order, trans_a, trans_b, m, n, k,
            alpha, a, lda, b, ldb, beta, c, ldc);
    else
        cblas_dgemm(
            order, trans_a, trans_b, m, n, k,
            alpha, a, lda, b, ldb, beta, c, ldc);
}

template <typename Precision>
inline void symmetrix_blas_gemv(
    const CBLAS_ORDER order,
    const CBLAS_TRANSPOSE trans,
    const int m,
    const int n,
    const Precision alpha,
    const Precision* a,
    const int lda,
    const Precision* x,
    const int incx,
    const Precision beta,
    Precision* y,
    const int incy)
{
    static_assert(std::is_same_v<Precision, float>
                  || std::is_same_v<Precision, double>);
    if constexpr (std::is_same_v<Precision, float>)
        cblas_sgemv(
            order, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    else
        cblas_dgemv(
            order, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

template <typename Precision>
inline void symmetrix_blas_copy(
    const int n,
    const Precision* x,
    const int incx,
    Precision* y,
    const int incy)
{
    static_assert(std::is_same_v<Precision, float>
                  || std::is_same_v<Precision, double>);
    if constexpr (std::is_same_v<Precision, float>)
        cblas_scopy(n, x, incx, y, incy);
    else
        cblas_dcopy(n, x, incx, y, incy);
}

template <typename Precision>
inline Precision symmetrix_blas_dot(
    const int n,
    const Precision* x,
    const int incx,
    const Precision* y,
    const int incy)
{
    static_assert(std::is_same_v<Precision, float>
                  || std::is_same_v<Precision, double>);
    if constexpr (std::is_same_v<Precision, float>)
        return cblas_sdot(n, x, incx, y, incy);
    else
        return cblas_ddot(n, x, incx, y, incy);
}
