#include <math.h>
#include <stdint.h>
#include <immintrin.h>

#include "types.h"

union Vec3 {
    struct {
        f64 x, y, z;
        f64 _pad = 0.0;
    };
    __m256d data;
};

union Mat3 {
    Vec3 data[3];
};

union AVX128d {
    __m128d avx;
    f64 el[2];
};

static inline Mat3 operator*(const f64 scalar, const Mat3& A) {
    __m256d s = _mm256_set1_pd(scalar);
    return Mat3{
        Vec3{ .data = _mm256_mul_pd(s, A.data[0].data) },
        Vec3{ .data = _mm256_mul_pd(s, A.data[1].data) },
        Vec3{ .data = _mm256_mul_pd(s, A.data[2].data) }
    };
}

static inline Mat3 operator+(const Mat3& A, const Mat3& B) {
    return Mat3{
        Vec3{ .data = _mm256_add_pd(A.data[0].data, B.data[0].data) },
        Vec3{ .data = _mm256_add_pd(A.data[1].data, B.data[1].data) },
        Vec3{ .data = _mm256_add_pd(A.data[2].data, B.data[2].data) }
    };
}

static inline Mat3 operator-(const Mat3& A, const Mat3& B) {
    return Mat3{
        Vec3{ .data = _mm256_sub_pd(A.data[0].data, B.data[0].data) },
        Vec3{ .data = _mm256_sub_pd(A.data[1].data, B.data[1].data) },
        Vec3{ .data = _mm256_sub_pd(A.data[2].data, B.data[2].data) }
    };
}

static inline Vec3 operator+(const Vec3& a, const Vec3& b) {
    return Vec3{ .data = _mm256_add_pd(a.data, b.data) };
}

static inline Vec3 operator-(const Vec3& a, const Vec3& b) {
    return Vec3{ .data = _mm256_sub_pd(a.data, b.data) };
}

static inline Vec3 operator*(const Vec3& a, const Vec3& b) {
    return Vec3{ .data = _mm256_mul_pd(a.data, b.data) };
}

static inline Vec3 operator*(const f64 scalar, const Vec3& v) {
    return Vec3{ .data = _mm256_mul_pd(_mm256_set1_pd(scalar), v.data) };
}

static inline Vec3 operator/(const Vec3& vec, const f64 denom) {
    return Vec3{ .data = _mm256_div_pd(vec.data, _mm256_set1_pd(denom))};
}

Mat3 identity() {
    return {
        Vec3{ .x = 1.0, .y = 0.0, .z = 0.0 },
        Vec3{ .x = 0.0, .y = 1.0, .z = 0.0 },
        Vec3{ .x = 0.0, .y = 0.0, .z = 1.0 }
    };
}

static inline f64 dot(const Vec3& a, const Vec3& b) {
    __m256d xy = _mm256_mul_pd(a.data, b.data);
    __m256d temp = _mm256_hadd_pd(xy, xy);
    AVX128d dp = { .avx = _mm_add_pd(
        _mm256_extractf128_pd(temp, 0),
        _mm256_extractf128_pd(temp, 1)
    ) };
    return dp.el[0];
}

static inline f64 norm(const Vec3& v) {
    return sqrt(dot(v, v));
}

static inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return {
        .x = a.y * b.z - a.z * b.y,
        .y = a.z * b.x - a.x * b.z,
        .z = a.x * b.y - a.y * b.x
    };
}

static inline Mat3 outer(const Vec3& a, const Vec3& b) {
    return { a.x * b, a.y * b, a.z * b };
}

static inline Mat3 transpose(const Mat3& A) {
    return {
        Vec3{ .x = A.data[0].x, .y = A.data[1].x, .z = A.data[2].x },
        Vec3{ .x = A.data[0].y, .y = A.data[1].y, .z = A.data[2].y },
        Vec3{ .x = A.data[0].z, .y = A.data[1].z, .z = A.data[2].z }
    };
}

static Vec3 matmul(const Mat3& A, const Vec3& v) {
    return Vec3{
        .x = dot(A.data[0], v),
        .y = dot(A.data[1], v),
        .z = dot(A.data[2], v)
    };
}

static Mat3 matmul(const Mat3& A, const Mat3& B) {
    // TODO: There should be a faster way to do this with fma-instructions and transpose A
    const Mat3 BT = transpose(B);
    return {
        Vec3{ .x = dot(A.data[0], BT.data[0]), .y = dot(A.data[0], BT.data[1]), .z = dot(A.data[0], BT.data[2]) },
        Vec3{ .x = dot(A.data[1], BT.data[0]), .y = dot(A.data[1], BT.data[1]), .z = dot(A.data[1], BT.data[2]) },
        Vec3{ .x = dot(A.data[2], BT.data[0]), .y = dot(A.data[2], BT.data[1]), .z = dot(A.data[2], BT.data[2]) }
    };
}

template<typename F, typename FPrime, typename... Types>
f64 newtonMethod(const f64 x0, F f, FPrime f_prime, Types... args) {
    f64 x, x_next = x0;
    do {
        x = x_next;
        x_next = x - f(x, args...) / f_prime(x, args...);
    } while (fabs(x - x_next) > 1e-6);
    return x_next;
}
