#ifndef DIAGO_PPCG_H_
#define DIAGO_PPCG_H_

#include "source_base/module_device/types.h"
#include "source_base/macros.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace hsolver
{

/**
 * @brief Gamma-point PPCG draft for source_hsolver.
 *
 * Notes:
 * 1. This file is intentionally aligned with the source_hsolver naming/interface style.
 * 2. The implementation below is a CPU-first draft and follows the uploaded ppcg_gamma.f90 logic.
 * 3. Projected matrices are treated as real-symmetric by using gamma-point real inner products.
 * 4. This is not a full MPI / band-group / GPU port of the original QE Fortran implementation.
 */
template <typename T = std::complex<double>, typename Device = base_device::DEVICE_CPU>
class DiagoPPCG final
{
  private:
    using Real = typename GetTypeReal<T>::type;

  public:
    using HPsiFunc = std::function<void(T*, T*, const int, const int)>;
    using SPsiFunc = std::function<void(T*, T*, const int, const int)>;

    DiagoPPCG(const Real& diag_thr,
              const int& diag_iter_max,
              const int& sbsize = 4,
              const int& rr_step = 2,
              const bool gamma_g0_real = true);

    double diag(const HPsiFunc& hpsi_func,
                const SPsiFunc& spsi_func,
                const int ld_psi,
                const int nband,
                const int dim,
                T* psi_in,
                Real* eigenvalue_in,
                const std::vector<double>& ethr_band,
                const Real* prec = nullptr);

  private:
    struct SmallSubspace
    {
        std::vector<Real> k;
        std::vector<Real> m;
        std::vector<Real> eval;
    };

    int ld_psi_ = 0;
    int n_band_ = 0;
    int n_dim_ = 0;
    int maxter_ = 0;
    int sbsize_ = 1;
    int rr_step_ = 1;

    Real diag_thr_ = static_cast<Real>(1.0e-8);
    bool gamma_g0_real_ = true;

    std::vector<T> hpsi_;
    std::vector<T> w_;
    std::vector<T> hw_;
    std::vector<T> p_;
    std::vector<T> hp_;

    static int idx(const int i, const int j, const int ld)
    {
        return i + j * ld;
    }

    void validate_input(T* psi_in, Real* eigenvalue_in, const Real* prec) const;
    void force_g0_real(T* x, const int ncol) const;

    void apply_h(const HPsiFunc& hpsi_func, T* psi_in, T* hpsi_out, const int ncol) const;
    void apply_s(const SPsiFunc& spsi_func, T* psi_in, T* spsi_out, const int ncol) const;

    Real gamma_dot(const T* x, const T* y) const;

    void gram(const T* a,
              const T* b,
              const int ncol_a,
              const int ncol_b,
              std::vector<Real>& out,
              const int ld_out) const;

    void copy_cols(const T* src, const std::vector<int>& cols, std::vector<T>& dst) const;
    void scatter_cols(T* dst, const std::vector<int>& cols, const std::vector<T>& src) const;

    void project_against(const T* basis,
                         const std::vector<int>& basis_cols,
                         std::vector<T>& x,
                         const std::vector<int>& x_cols) const;

    void divide_by_preconditioner(const std::vector<int>& active_cols,
                                  const Real* prec,
                                  std::vector<T>& x) const;

    void lock_epairs(const std::vector<T>& residual,
                     const std::vector<double>& ethr_band,
                     std::vector<int>& active_cols) const;

    void build_small_subspace(const T* psi,
                              const std::vector<int>& cols,
                              const bool use_p,
                              SmallSubspace& subspace) const;

    void solve_small_generalized(const int dim, SmallSubspace& subspace) const;

    void update_one_block(T* psi,
                          const std::vector<int>& cols,
                          const int l,
                          const bool use_p,
                          const SmallSubspace& subspace) const;

    void chol_qr_active(T* psi, const std::vector<int>& active_cols) const;
    void right_solve_upper_real(const std::vector<Real>& r,
                                const int n,
                                std::vector<T>& x) const;

    void rayleigh_ritz(T* psi,
                       Real* eigenvalue,
                       std::vector<int>& active_cols,
                       const std::vector<double>& ethr_band);

    Real trace_of_active_projected(const T* psi, const std::vector<int>& active_cols) const;
};

} // namespace hsolver

#endif // DIAGO_PPCG_H_
