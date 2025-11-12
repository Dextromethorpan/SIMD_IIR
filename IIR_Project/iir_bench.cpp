#include <immintrin.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

struct Args {
    int S = 200000; // number of samples
    int minN = 2, maxN = 16, step = 2;
    int trials = 5;
};

Args parse(int argc, char** argv){
    Args a;
    for (int i=1;i<argc;++i){
        std::string s(argv[i]);
        auto nexti = [&](int def)->int{ if(i+1<argc) return std::stoi(argv[++i]); return def; };
        if (s=="--samples") a.S = nexti(a.S);
        else if (s=="--minN") a.minN = nexti(a.minN);
        else if (s=="--maxN") a.maxN = nexti(a.maxN);
        else if (s=="--step")  a.step  = nexti(a.step);
        else if (s=="--trials") a.trials = nexti(a.trials);
    }
    return a;
}

// Generate random stable-ish feedback taps
std::vector<float> random_a(int N, float rho=0.6f){
    std::mt19937 rng(12345);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> a(N);
    for (int i=0;i<N;++i) a[i] = nd(rng) * (rho / std::max(1, N));
    return a;
}

std::vector<float> randn(int S){
    std::mt19937 rng(54321);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x(S);
    for (int i=0;i<S;++i) x[i] = nd(rng);
    return x;
}

// Scalar baseline
void iir_scalar(const std::vector<float>& x, const std::vector<float>& a, std::vector<float>& y){
    const int N = (int)a.size();
    const int S = (int)x.size();
    std::fill(y.begin(), y.end(), 0.f);
    for (int n=0; n<S; ++n){
        float acc = x[n];
        for (int i=1; i<=N; ++i){
            if (n - i >= 0) acc += a[i-1] * y[n-i];
        }
        y[n] = acc;
    }
}

std::vector<float> fused_c(const std::vector<float>& a, int p){
    std::vector<float> c(p, 0.f);
    c[0] = 1.f;
    for (int k=1;k<p;++k){
        float s = 0.f;
        for (int i=1;i<=k;++i){
            if (i-1 < (int)a.size()) s += a[i-1] * c[k - i];
        }
        c[k] = s;
    }
    return c;
}

// SSE (p=4)
void iir_sse_fused(const std::vector<float>& x, const std::vector<float>& a, std::vector<float>& y){
    const int p = 4;
    const int S = (int)x.size();
    const int N = (int)a.size();
    std::fill(y.begin(), y.end(), 0.f);
    auto c = fused_c(a, p);

    int n = 0;
    while (n < S){
        int block = std::min(p, S - n);
        float vtmp[4] = {0,0,0,0};
        for (int k=0;k<block;++k) vtmp[k] += x[n+k];
        __m128 v = _mm_loadu_ps(vtmp);

        for (int i=1; i<=p; ++i){
            float src_a[4] = {0,0,0,0};
            int tail = p - i;
            if (tail > 0){
                for (int t=0;t<tail;++t){
                    int idx = n - tail + t;
                    src_a[t] = (idx >= 0) ? y[idx] : 0.f;
                }
            }
            float fill = ((p - i) < p) ? vtmp[p - i] : 0.f;
            for (int t=tail; t<p; ++t) src_a[t] = fill;
            __m128 src = _mm_loadu_ps(src_a);

            float s_a[4] = {0,0,0,0};
            if (tail > 0){
                int aidx = tail - 1;
                float aval = (aidx >=0 && aidx < N) ? a[aidx] : 0.f;
                for (int t=0;t<tail;++t) s_a[t] = aval;
            }
            for (int k=1;k<=i;++k){
                float ck = (k < (int)c.size()) ? c[k] : 0.f;
                s_a[p - k] = ck;
            }
            __m128 svec = _mm_loadu_ps(s_a);
            v = _mm_add_ps(v, _mm_mul_ps(src, svec));
            _mm_storeu_ps(vtmp, v);
        }
        for (int k=0;k<block;++k) y[n+k] = vtmp[k];
        n += block;
    }
}

// AVX (p=8)
void iir_avx_fused(const std::vector<float>& x, const std::vector<float>& a, std::vector<float>& y){
    const int p = 8;
    const int S = (int)x.size();
    const int N = (int)a.size();
    std::fill(y.begin(), y.end(), 0.f);
    auto c = fused_c(a, p);

    int n = 0;
    while (n < S){
        int block = std::min(p, S - n);
        float vtmp[8] = {0,0,0,0,0,0,0,0};
        for (int k=0;k<block;++k) vtmp[k] += x[n+k];
        __m256 v = _mm256_loadu_ps(vtmp);

        for (int i=1; i<=p; ++i){
            float src_a[8] = {0,0,0,0,0,0,0,0};
            int tail = p - i;
            if (tail > 0){
                for (int t=0;t<tail;++t){
                    int idx = n - tail + t;
                    src_a[t] = (idx >= 0) ? y[idx] : 0.f;
                }
            }
            float fill = ((p - i) < p) ? vtmp[p - i] : 0.f;
            for (int t=tail; t<p; ++t) src_a[t] = fill;
            __m256 src = _mm256_loadu_ps(src_a);

            float s_a[8] = {0,0,0,0,0,0,0,0};
            if (tail > 0){
                int aidx = tail - 1;
                float aval = (aidx >=0 && aidx < N) ? a[aidx] : 0.f;
                for (int t=0;t<tail;++t) s_a[t] = aval;
            }
            for (int k=1;k<=i;++k){
                s_a[p - k] = (k < (int)c.size()) ? c[k] : 0.f;
            }
            __m256 svec = _mm256_loadu_ps(s_a);
            v = _mm256_add_ps(v, _mm256_mul_ps(src, svec));
            _mm256_storeu_ps(vtmp, v);
        }
        for (int k=0;k<block;++k) y[n+k] = vtmp[k];
        n += block;
    }
}

using Clock = std::chrono::high_resolution_clock;

int main(int argc, char** argv){
    auto args = parse(argc, argv);
    std::FILE* f = std::fopen("bench_results.csv", "w");
    std::fprintf(f, "version,N_taps,S,trial,ns_per_sample_per_tap\n");

    for (int N=args.minN; N<=args.maxN; N+=args.step){
        auto a = random_a(N, 0.6f);
        auto x = randn(args.S);
        std::vector<float> y(args.S, 0.f);

        for (int t=0; t<args.trials; ++t){
            auto t0 = Clock::now();
            iir_scalar(x, a, y);
            auto t1 = Clock::now();
            double dt = std::chrono::duration<double>(t1 - t0).count();
            double nspt = (dt / args.S) * 1e9 / std::max(1, N);
            std::fprintf(f, "scalar,%d,%d,%d,%.9f\n", N, args.S, t, nspt);

            t0 = Clock::now();
            iir_sse_fused(x, a, y);
            t1 = Clock::now();
            dt = std::chrono::duration<double>(t1 - t0).count();
            nspt = (dt / args.S) * 1e9 / std::max(1, N);
            std::fprintf(f, "sse,%d,%d,%d,%.9f\n", N, args.S, t, nspt);

            t0 = Clock::now();
            iir_avx_fused(x, a, y);
            t1 = Clock::now();
            dt = std::chrono::duration<double>(t1 - t0).count();
            nspt = (dt / args.S) * 1e9 / std::max(1, N);
            std::fprintf(f, "avx,%d,%d,%d,%.9f\n", N, args.S, t, nspt);
        }
    }

    std::fclose(f);
    std::cerr << "Wrote bench_results.csv\n";
    return 0;
}
