#include "source_hsolver/diago_ppcg.h"

#include <limits>
#include <stdexcept>
#include <vector>

extern "C"
{
void dsyevd_(const char* jobz, const char* uplo,
             const int* n, double* a, const int* lda, double* w,
             double* work, const int* lwork, int* iwork,
             const int* liwork, int* info);

void ssyevd_(const char* jobz, const char* uplo,
             const int* n, float* a, const int* lda, float* w,
             float* work, const int* lwork, int* iwork,
             const int* liwork, int* info);

void dsygvd_(const int* itype, const char* jobz, const char* uplo,
             const int* n, double* a, const int* lda, double* b,
             const int* ldb, double* w, double* work, const int* lwork,
             int* iwork, const int* liwork, int* info);

void ssygvd_(const int* itype, const char* jobz, const char* uplo,
             const int* n, float* a, const int* lda, float* b,
             const int* ldb, float* w, float* work, const int* lwork,
             int* iwork, const int* liwork, int* info);

void dpotrf_(const char* uplo, const int* n, double* a, const int* lda, int* info);
void spotrf_(const char* uplo, const int* n, float* a, const int* lda, int* info);
}

namespace hsolver
{
namespace
{

template <typename Real>
struct LapackReal;

template <>
struct LapackReal<double>
{
    static void syevd(const int n, double* a, double* w)
    {
        const char jobz = 'V';
        const char uplo = 'U';
        const int lda = n;
        int info = 0;
        const int lwork = std::max(1, 1 + 6 * n + 2 * n * n);
        const int liwork = std::max(1, 3 + 5 * n);
        std::vector<double> work(lwork, 0.0);
        std::vector<int> iwork(liwork, 0);
        dsyevd_(&jobz, &uplo, &n, a, &lda, w,
                work.data(), &lwork, iwork.data(), &liwork, &info);
        if (info != 0)
        {
            throw std::runtime_error("DiagoPPCG: dsyevd failed.");
        }
    }

    static void sygvd(const int n, double* a, double* b, double* w)
    {
        const int itype = 1;
        const char jobz = 'V';
        const char uplo = 'U';
        const int lda = n;
        const int ldb = n;
        int info = 0;
        const int lwork = std::max(1, 1 + 18 * n + 10 * n * n);
        const int liwork = std::max(1, 3 + 10 * n);
        std::vector<double> work(lwork, 0.0);
        std::vector<int> iwork(liwork, 0);
        dsygvd_(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w,
                work.data(), &lwork, iwork.data(), &liwork, &info);
        if (info != 0)
        {
            throw std::runtime_error("DiagoPPCG: dsygvd failed.");
        }
    }

    static void potrf(const int n, double* a)
    {
        const char uplo = 'U';
        const int lda = n;
        int info = 0;
        dpotrf_(&uplo, &n, a, &lda, &info);
        if (info != 0)
        {
            throw std::runtime_error("DiagoPPCG: dpotrf failed.");
        }
    }
};

template <>
struct LapackReal<float>
{
    static void syevd(const int n, float* a, float* w)
    {
        const char jobz = 'V';
        const char uplo = 'U';
        const int lda = n;
        int info = 0;
        const int lwork = std::max(1, 1 + 6 * n + 2 * n * n);
        const int liwork = std::max(1, 3 + 5 * n);
        std::vector<float> work(lwork, 0.0f);
        std::vector<int> iwork(liwork, 0);
        ssyevd_(&jobz, &uplo, &n, a, &lda, w,
                work.data(), &lwork, iwork.data(), &liwork, &info);
        if (info != 0)
        {
            throw std::runtime_error("DiagoPPCG: ssyevd failed.");
        }
    }

    static void sygvd(const int n, float* a, float* b, float* w)
    {
        const int itype = 1;
        const char jobz = 'V';
        const char uplo = 'U';
        const int lda = n;
        const int ldb = n;
        int info = 0;
        const int lwork = std::max(1, 1 + 18 * n + 10 * n * n);
        const int liwork = std::max(1, 3 + 10 * n);
        std::vector<float> work(lwork, 0.0f);
        std::vector<int> iwork(liwork, 0);
        ssygvd_(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w,
                work.data(), &lwork, iwork.data(), &liwork, &info);
        if (info != 0)
        {
            throw std::runtime_error("DiagoPPCG: ssygvd failed.");
        }
    }

    static void potrf(const int n, float* a)
    {
        const char uplo = 'U';
        const int lda = n;
        int info = 0;
        spotrf_(&uplo, &n, a, &lda, &info);
        if (info != 0)
        {
            throw std::runtime_error("DiagoPPCG: spotrf failed.");
        }
    }
};

template <typename T>
inline T zero_value()
{
    return T(0);
}

template <typename T>
inline void set_zero(std::vector<T>& x)
{
    std::fill(x.begin(), x.end(), zero_value<T>());
}

} // namespace

template <typename T, typename Device>
DiagoPPCG<T, Device>::DiagoPPCG(const Real& diag_thr,
                                const int& diag_iter_max,
                                const int& sbsize,
                                const int& rr_step,
                                const bool gamma_g0_real)
    : maxter_(diag_iter_max),
      sbsize_(std::max(1, sbsize)),
      rr_step_(std::max(1, rr_step)),
      diag_thr_(std::max(diag_thr, static_cast<Real>(1.0e-14))),
      gamma_g0_real_(gamma_g0_real)
{
}

template <typename T, typename Device>
void DiagoPPCG<T, Device>::validate_input(T* psi_in, Real* eigenvalue_in, const Real* prec) const
{
    if (psi_in == nullptr || eigenvalue_in == nullptr)
    {
        throw std::invalid_argument("DiagoPPCG: psi/eigenvalue pointer is null.");
    }
    if (prec == nullptr)
    {
        throw std::invalid_argument("DiagoPPCG: precondition pointer is null.");
    }
    if (ld_psi_ <= 0 || n_band_ <= 0 || n_dim_ <= 0)
    {
        throw std::invalid_argument("DiagoPPCG: invalid dimensions.");
    }
    if (n_dim_ > ld_psi_)
    {
        throw std::invalid_argument("DiagoPPCG: dim must not exceed ld_psi.");
    }
}

template <typename T, typename Device>
void DiagoPPCG<T, Device>::force_g0_real(T* x, const int ncol) const
{
    if (!gamma_g0_real_ || n_dim_ <= 0)
    {
        return;
    }
    for (int j = 0; j < ncol; ++j)
    {
        x[idx(0, j, ld_psi_)] = T(std::real(x[idx(0, j, ld_psi_)]), 0.0);
    }
}

template <typename T, typename Device>
void DiagoPPCG<T, Device>::apply_h(const HPsiFunc& hpsi_func, T* psi_in, T* hpsi_out, const int ncol) const
{
    hpsi_func(psi_in, hpsi_out, ld_psi_, ncol);
}

template <typename T, typename Device>
void DiagoPPCG<T, Device>::apply_s(const SPsiFunc& spsi_func, T* psi_in, T* spsi_out, const int ncol) const
{
    if (spsi_func)
    {
        spsi_func(psi_in, spsi_out, ld_psi_, ncol);
    }
    else
    {
        for (int j = 0; j < ncol; ++j)
        {
            std::copy(psi_in + j * ld_psi_, psi_in + (j + 1) * ld_psi_, spsi_out + j * ld_psi_);
        }
    }
}

template <typename T, typename Device>
typename DiagoPPCG<T, Device>::Real DiagoPPCG<T, Device>::gamma_dot(const T* x, const T* y) const
{
    Real acc = 0;
    for (int i = 0; i < n_dim_; ++i)
    {
        acc += static_cast<Real>(std::real(std::conj(x[i]) * y[i]));
    }
    return acc;
}

template <typename T, typename Device>
void DiagoPPCG<T, Device>::gram(const T* a,
                                const T* b,
                                const int ncol_a,
                                const int ncol_b,
                                std::vector<Real>& out,
                                const int ld_out) const
{
    out.assign(ld_out * ncol_b, static_cast<Real>(0));
    for (int jb = 0; jb < ncol_b; ++jb)
    {
        for (int ia = 0; ia < ncol_a; ++ia)
        {
            out[ia + jb * ld_out] = gamma_dot(a + ia * ld_psi_, b + jb * ld_psi_);
        }
    }
}

template <typename T, typename Device>
void DiagoPPCG<T, Device>::copy_cols(const T* src, const std::vector<int>& cols, std::vector<T>& dst) const
{
    dst.assign(ld_psi_ * cols.size(), zero_value<T>());
    for (int j = 0; j < static_cast<int>(cols.size()); ++j)
    {
        const int c = cols[j];
        std::copy(src + c * ld_psi_, src + c * ld_psi_ + ld_psi_, dst.begin() + j * ld_psi_);
    }
}

template <typename T, typename Device>
void DiagoPPCG<T, Device>::scatter_cols(T* dst, const std::vector<int>& cols, const std::vector<T>& src) const
{
    for (int j = 0; j < static_cast<int>(cols.size()); ++j)
    {
        const int c = cols[j];
        std::copy(src.begin() + j * ld_psi_, src.begin() + (j + 1) * ld_psi_, dst + c * ld_psi_);
    }
}

template <typename T, typename Device>
void DiagoPPCG<T, Device>::project_against(const T* basis,
                                           const std::vector<int>& basis_cols,
                                           std::vector<T>& x,
                                           const std::vector<int>& x_cols) const
{
    if (basis_cols.empty() || x_cols.empty())
    {
        return;
    }

    std::vector<T> basis_block;
    copy_cols(basis, basis_cols, basis_block);

    std::vector<T> x_block;
    copy_cols(x.data(), x_cols, x_block);

    std::vector<Real> g;
    gram(basis_block.data(),
         x_block.data(),
         static_cast<int>(basis_cols.size()),
         static_cast<int>(x_cols.size()),
         g,
         static_cast<int>(basis_cols.size()));

    for (int j = 0; j < static_cast<int>(x_cols.size()); ++j)
    {
        const int c = x_cols[j];
        for (int i = 0; i < static_cast<int>(basis_cols.size()); ++i)
        {
            const int bc = basis_cols[i];
            const Real coeff = g[i + j * static_cast<int>(basis_cols.size())];
            for (int ig = 0; ig < n_dim_; ++ig)
            {
                x[idx(ig, c, ld_psi_)] -= basis[idx(ig, bc, ld_psi_)] * coeff;
            }
        }
    }
}

template <typename T, typename Device>
void DiagoPPCG<T, Device>::divide_by_preconditioner(const std::vector<int>& active_cols,
                                                    const Real* prec,
                                                    std::vector<T>& x) const
{
    for (const int c : active_cols)
    {
        for (int ig = 0; ig < n_dim_; ++ig)
        {
            x[idx(ig, c, ld_psi_)] /= std::max(prec[ig], static_cast<Real>(1.0e-12));
        }
    }
}

template <typename T, typename Device>
void DiagoPPCG<T, Device>::lock_epairs(const std::vector<T>& residual,
                                       const std::vector<double>& ethr_band,
                                       std::vector<int>& active_cols) const
{
    active_cols.clear();
    for (int j = 0; j < n_band_; ++j)
    {
        Real nrm2 = 0;
        for (int ig = 0; ig < n_dim_; ++ig)
        {
            nrm2 += static_cast<Real>(std::norm(residual[idx(ig, j, ld_psi_)]));
        }
        const Real rnrm = std::sqrt(std::max(nrm2, static_cast<Real>(0)));
        const Real thr = std::sqrt(std::max(static_cast<Real>(ethr_band[j]), diag_thr_));
        if (rnrm > thr)
        {
            active_cols.push_back(j);
        }
    }
}

template <typename T, typename Device>
void DiagoPPCG<T, Device>::build_small_subspace(const T* psi,
                                                const std::vector<int>& cols,
                                                const bool use_p,
                                                SmallSubspace& subspace) const
{
    const int l = static_cast<int>(cols.size());
    const int nblk = use_p ? 3 : 2;
    const int dim = nblk * l;
    subspace.k.assign(dim * dim, static_cast<Real>(0));
    subspace.m.assign(dim * dim, static_cast<Real>(0));
    subspace.eval.assign(dim, static_cast<Real>(0));

    std::vector<T> psi_l, hpsi_l, w_l, hw_l, p_l, hp_l;
    copy_cols(psi, cols, psi_l);
    copy_cols(hpsi_.data(), cols, hpsi_l);
    copy_cols(w_.data(), cols, w_l);
    copy_cols(hw_.data(), cols, hw_l);
    if (use_p)
    {
        copy_cols(p_.data(), cols, p_l);
        copy_cols(hp_.data(), cols, hp_l);
    }

    auto fill_sym_block = [&](const std::vector<T>& a,
                              const std::vector<T>& b,
                              const int r0,
                              const int c0,
                              std::vector<Real>& mat)
    {
        std::vector<Real> g;
        gram(a.data(), b.data(), l, l, g, l);
        for (int j = 0; j < l; ++j)
        {
            for (int i = 0; i < l; ++i)
            {
                mat[(r0 + i) + (c0 + j) * dim] = g[i + j * l];
                mat[(c0 + j) + (r0 + i) * dim] = g[i + j * l];
            }
        }
    };

    fill_sym_block(psi_l, hpsi_l, 0, 0, subspace.k);
    fill_sym_block(psi_l, psi_l, 0, 0, subspace.m);

    fill_sym_block(w_l, hw_l, l, l, subspace.k);
    fill_sym_block(w_l, w_l, l, l, subspace.m);

    fill_sym_block(psi_l, hw_l, 0, l, subspace.k);
    fill_sym_block(psi_l, w_l, 0, l, subspace.m);

    if (use_p)
    {
        fill_sym_block(p_l, hp_l, 2 * l, 2 * l, subspace.k);
        fill_sym_block(p_l, p_l, 2 * l, 2 * l, subspace.m);

        fill_sym_block(psi_l, hp_l, 0, 2 * l, subspace.k);
        fill_sym_block(psi_l, p_l, 0, 2 * l, subspace.m);

        fill_sym_block(w_l, hp_l, l, 2 * l, subspace.k);
        fill_sym_block(w_l, p_l, l, 2 * l, subspace.m);
    }
}

template <typename T, typename Device>
void DiagoPPCG<T, Device>::solve_small_generalized(const int dim, SmallSubspace& subspace) const
{
    LapackReal<Real>::sygvd(dim, subspace.k.data(), subspace.m.data(), subspace.eval.data());
}

template <typename T, typename Device>
void DiagoPPCG<T, Device>::update_one_block(T* psi,
                                            const std::vector<int>& cols,
                                            const int l,
                                            const bool use_p,
                                            const SmallSubspace& subspace) const
{
    const int dim = (use_p ? 3 : 2) * l;
    const Real* eigvec = subspace.k.data();

    std::vector<T> psi_l, hpsi_l, w_l, hw_l, p_l, hp_l;
    copy_cols(psi, cols, psi_l);
    copy_cols(hpsi_.data(), cols, hpsi_l);
    copy_cols(w_.data(), cols, w_l);
    copy_cols(hw_.data(), cols, hw_l);
    if (use_p)
    {
        copy_cols(p_.data(), cols, p_l);
        copy_cols(hp_.data(), cols, hp_l);
    }

    std::vector<T> psi_new(ld_psi_ * l, zero_value<T>());
    std::vector<T> hpsi_new(ld_psi_ * l, zero_value<T>());
    std::vector<T> p_new(ld_psi_ * l, zero_value<T>());
    std::vector<T> hp_new(ld_psi_ * l, zero_value<T>());

    for (int j = 0; j < l; ++j)
    {
        for (int i = 0; i < l; ++i)
        {
            const Real cpsi = eigvec[i + j * dim];
            const Real cw = eigvec[(l + i) + j * dim];

            for (int ig = 0; ig < n_dim_; ++ig)
            {
                psi_new[idx(ig, j, ld_psi_)] += psi_l[idx(ig, i, ld_psi_)] * cpsi
                                              + w_l[idx(ig, i, ld_psi_)] * cw;
                hpsi_new[idx(ig, j, ld_psi_)] += hpsi_l[idx(ig, i, ld_psi_)] * cpsi
                                               + hw_l[idx(ig, i, ld_psi_)] * cw;
                p_new[idx(ig, j, ld_psi_)] += w_l[idx(ig, i, ld_psi_)] * cw;
                hp_new[idx(ig, j, ld_psi_)] += hw_l[idx(ig, i, ld_psi_)] * cw;
            }

            if (use_p)
            {
                const Real cp = eigvec[(2 * l + i) + j * dim];
                for (int ig = 0; ig < n_dim_; ++ig)
                {
                    psi_new[idx(ig, j, ld_psi_)] += p_l[idx(ig, i, ld_psi_)] * cp;
                    hpsi_new[idx(ig, j, ld_psi_)] += hp_l[idx(ig, i, ld_psi_)] * cp;
                    p_new[idx(ig, j, ld_psi_)] += p_l[idx(ig, i, ld_psi_)] * cp;
                    hp_new[idx(ig, j, ld_psi_)] += hp_l[idx(ig, i, ld_psi_)] * cp;
                }
            }
        }
    }

    scatter_cols(psi, cols, psi_new);
    scatter_cols(hpsi_.data(), cols, hpsi_new);
    scatter_cols(p_.data(), cols, p_new);
    scatter_cols(hp_.data(), cols, hp_new);
}

template <typename T, typename Device>
void DiagoPPCG<T, Device>::right_solve_upper_real(const std::vector<Real>& r,
                                                  const int n,
                                                  std::vector<T>& x) const
{
    std::vector<T> b = x;
    for (int row = 0; row < n_dim_; ++row)
    {
        for (int j = 0; j < n; ++j)
        {
            T v = b[idx(row, j, ld_psi_)];
            for (int k = 0; k < j; ++k)
            {
                v -= x[idx(row, k, ld_psi_)] * r[k + j * n];
            }
            x[idx(row, j, ld_psi_)] = v / r[j + j * n];
        }
    }
}

template <typename T, typename Device>
void DiagoPPCG<T, Device>::chol_qr_active(T* psi, const std::vector<int>& active_cols) const
{
    if (active_cols.empty())
    {
        return;
    }

    const int nact = static_cast<int>(active_cols.size());
    std::vector<T> psi_a, hpsi_a;
    copy_cols(psi, active_cols, psi_a);
    copy_cols(hpsi_.data(), active_cols, hpsi_a);

    std::vector<Real> s(nact * nact, static_cast<Real>(0));
    gram(psi_a.data(), psi_a.data(), nact, nact, s, nact);

    LapackReal<Real>::potrf(nact, s.data());

    right_solve_upper_real(s, nact, psi_a);
    right_solve_upper_real(s, nact, hpsi_a);

    scatter_cols(psi, active_cols, psi_a);
    scatter_cols(hpsi_.data(), active_cols, hpsi_a);
}

template <typename T, typename Device>
void DiagoPPCG<T, Device>::rayleigh_ritz(T* psi,
                                         Real* eigenvalue,
                                         std::vector<int>& active_cols,
                                         const std::vector<double>& ethr_band)
{
    std::vector<Real> hsub(n_band_ * n_band_, static_cast<Real>(0));
    gram(psi, hpsi_.data(), n_band_, n_band_, hsub, n_band_);

    std::vector<Real> eval(n_band_, static_cast<Real>(0));
    LapackReal<Real>::syevd(n_band_, hsub.data(), eval.data());

    std::vector<T> psi_old(psi, psi + ld_psi_ * n_band_);
    std::vector<T> hpsi_old = hpsi_;

    std::fill(psi, psi + ld_psi_ * n_band_, zero_value<T>());
    set_zero(hpsi_);

    for (int j = 0; j < n_band_; ++j)
    {
        for (int i = 0; i < n_band_; ++i)
        {
            const Real c = hsub[i + j * n_band_];
            for (int ig = 0; ig < n_dim_; ++ig)
            {
                psi[idx(ig, j, ld_psi_)] += psi_old[idx(ig, i, ld_psi_)] * c;
                hpsi_[idx(ig, j, ld_psi_)] += hpsi_old[idx(ig, i, ld_psi_)] * c;
            }
        }
        eigenvalue[j] = eval[j];
    }

    set_zero(w_);
    for (int j = 0; j < n_band_; ++j)
    {
        for (int ig = 0; ig < n_dim_; ++ig)
        {
            w_[idx(ig, j, ld_psi_)] = hpsi_[idx(ig, j, ld_psi_)] - psi[idx(ig, j, ld_psi_)] * eigenvalue[j];
        }
    }

    lock_epairs(w_, ethr_band, active_cols);
}

template <typename T, typename Device>
typename DiagoPPCG<T, Device>::Real
DiagoPPCG<T, Device>::trace_of_active_projected(const T* psi, const std::vector<int>& active_cols) const
{
    if (active_cols.empty())
    {
        return static_cast<Real>(0);
    }

    std::vector<T> psi_a, hpsi_a;
    copy_cols(psi, active_cols, psi_a);
    copy_cols(hpsi_.data(), active_cols, hpsi_a);

    const int nact = static_cast<int>(active_cols.size());
    std::vector<Real> g(nact * nact, static_cast<Real>(0));
    gram(psi_a.data(), hpsi_a.data(), nact, nact, g, nact);

    Real tr = 0;
    for (int i = 0; i < nact; ++i)
    {
        tr += g[i + i * nact];
    }
    return tr;
}

template <typename T, typename Device>
double DiagoPPCG<T, Device>::diag(const HPsiFunc& hpsi_func,
                                  const SPsiFunc&,
                                  const int ld_psi,
                                  const int nband,
                                  const int dim,
                                  T* psi_in,
                                  Real* eigenvalue_in,
                                  const std::vector<double>& ethr_band,
                                  const Real* prec)
{
    ld_psi_ = ld_psi;
    n_band_ = nband;
    n_dim_ = dim;

    validate_input(psi_in, eigenvalue_in, prec);

    hpsi_.assign(ld_psi_ * n_band_, zero_value<T>());
    w_.assign(ld_psi_ * n_band_, zero_value<T>());
    hw_.assign(ld_psi_ * n_band_, zero_value<T>());
    p_.assign(ld_psi_ * n_band_, zero_value<T>());
    hp_.assign(ld_psi_ * n_band_, zero_value<T>());

    std::vector<int> all_cols(n_band_);
    std::iota(all_cols.begin(), all_cols.end(), 0);

    force_g0_real(psi_in, n_band_);
    apply_h(hpsi_func, psi_in, hpsi_.data(), n_band_);

    std::vector<Real> g(n_band_ * n_band_, static_cast<Real>(0));
    gram(psi_in, hpsi_.data(), n_band_, n_band_, g, n_band_);

    for (int j = 0; j < n_band_; ++j)
    {
        eigenvalue_in[j] = g[j + j * n_band_];
        for (int ig = 0; ig < n_dim_; ++ig)
        {
            T sum = zero_value<T>();
            for (int i = 0; i < n_band_; ++i)
            {
                sum += psi_in[idx(ig, i, ld_psi_)] * g[i + j * n_band_];
            }
            w_[idx(ig, j, ld_psi_)] = hpsi_[idx(ig, j, ld_psi_)] - sum;
        }
    }

    std::vector<int> active_cols;
    lock_epairs(w_, ethr_band, active_cols);

    Real trG = trace_of_active_projected(psi_in, active_cols);
    Real trdif = static_cast<Real>(-1);
    double avg_iter = 1.0;
    int iter = 1;

    while (!active_cols.empty() && iter <= maxter_)
    {
        const int nact = static_cast<int>(active_cols.size());
        const int nsb = std::max(1, (nact + sbsize_ - 1) / sbsize_);
        const Real trtol = diag_thr_ * std::sqrt(static_cast<Real>(nact));

        divide_by_preconditioner(active_cols, prec, w_);
        project_against(psi_in, all_cols, w_, active_cols);

        std::vector<T> w_active;
        copy_cols(w_.data(), active_cols, w_active);
        force_g0_real(w_active.data(), nact);
        std::vector<T> hw_active(ld_psi_ * nact, zero_value<T>());
        apply_h(hpsi_func, w_active.data(), hw_active.data(), nact);
        scatter_cols(hw_.data(), active_cols, hw_active);

        avg_iter += static_cast<double>(nact) / static_cast<double>(n_band_);

        const bool use_p = (iter != 1);
        if (use_p)
        {
            project_against(psi_in, active_cols, p_, active_cols);
        }

        for (int isb = 0; isb < nsb; ++isb)
        {
            const int i0 = isb * sbsize_;
            const int l = std::min(sbsize_, nact - i0);
            std::vector<int> cols(active_cols.begin() + i0, active_cols.begin() + i0 + l);

            SmallSubspace subspace;
            build_small_subspace(psi_in, cols, use_p, subspace);
            solve_small_generalized((use_p ? 3 : 2) * l, subspace);
            update_one_block(psi_in, cols, l, use_p, subspace);
        }

        if (iter % rr_step_ == 0)
        {
            rayleigh_ritz(psi_in, eigenvalue_in, active_cols, ethr_band);
            trdif = static_cast<Real>(-1);
            trG = 0;
            for (const int c : active_cols)
            {
                trG += eigenvalue_in[c];
            }
        }
        else
        {
            chol_qr_active(psi_in, active_cols);

            std::vector<T> psi_a, hpsi_a;
            copy_cols(psi_in, active_cols, psi_a);
            copy_cols(hpsi_.data(), active_cols, hpsi_a);

            const int na = static_cast<int>(active_cols.size());
            std::vector<Real> ga(na * na, static_cast<Real>(0));
            gram(psi_a.data(), hpsi_a.data(), na, na, ga, na);

            set_zero(w_);
            for (int ja = 0; ja < na; ++ja)
            {
                for (int ig = 0; ig < n_dim_; ++ig)
                {
                    T sum = zero_value<T>();
                    for (int ia = 0; ia < na; ++ia)
                    {
                        sum += psi_a[idx(ig, ia, ld_psi_)] * ga[ia + ja * na];
                    }
                    w_[idx(ig, active_cols[ja], ld_psi_)] = hpsi_a[idx(ig, ja, ld_psi_)] - sum;
                }
            }

            Real trG1 = 0;
            for (int i = 0; i < na; ++i)
            {
                trG1 += ga[i + i * na];
            }

            trdif = std::abs(trG1 - trG);
            trG = trG1;

            if (trdif >= 0 && trdif <= trtol)
            {
                break;
            }
        }

        ++iter;
    }

    if ((iter - 1) % rr_step_ != 0)
    {
        rayleigh_ritz(psi_in, eigenvalue_in, active_cols, ethr_band);
    }

    return avg_iter;
}

template class DiagoPPCG<std::complex<float>, base_device::DEVICE_CPU>;
template class DiagoPPCG<std::complex<double>, base_device::DEVICE_CPU>;

} // namespace hsolver
