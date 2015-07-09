#include <bits/stdc++.h>
#include <cmath>
#include "sys/time.h"
#include "omp.h"
#include "float.h"
#include "xmmintrin.h"

#define SIMD_WIDTH 4

#ifdef MNIST
#define NUM_POINTS 70000
#define DIMENSION 784
#define ROUNDED_DIMENSION 1024
#define NUM_CLUSTER 10
#define ROUNDED_CLUSTER 16
#define DATA "mnist"
#else
#ifdef THREE_D
#define NUM_POINTS 434874
#define DIMENSION 3
#define ROUNDED_DIMENSION 4
#define NUM_CLUSTER 5
#define ROUNDED_CLUSTER 8
#define DATA "3d_pro_5"
#else
#ifdef CIFAR
#define NUM_POINTS 6000
#define DIMENSION 3072
#define ROUNDED_DIMENSION 4096
#define NUM_CLUSTER 10
#define ROUNDED_CLUSTER 16
#define DATA "cifar"
#else
#ifdef BIRCH1
#define NUM_POINTS 100000
#define DIMENSION 2
#define ROUNDED_DIMENSION 2
#define NUM_CLUSTER 100
#define ROUNDED_CLUSTER 128
#define DATA "birch1"
#else
#ifdef BIRCH2
#define NUM_POINTS 100000
#define DIMENSION 2
#define ROUNDED_DIMENSION 2
#define NUM_CLUSTER 100
#define ROUNDED_CLUSTER 128
#define DATA "birch2"
#else
#ifdef BIRCH3
#define NUM_POINTS 100000
#define DIMENSION 2
#define ROUNDED_DIMENSION 2
#define NUM_CLUSTER 100
#define ROUNDED_CLUSTER 128
#define DATA "birch3"
#endif
#endif
#endif
#endif
#endif
#endif

using namespace std;

typedef float Point __attribute__((vector_size(ROUNDED_DIMENSION*sizeof(float))));
typedef float Distances __attribute__((vector_size(ROUNDED_CLUSTER*sizeof(float))));


static inline float distance(Point, Point);
static inline int sample_from_distribution(vector<float> &, int, int, float);
static inline float get_time_diff(struct timeval, struct timeval);

static inline string mean(float*,int);
static inline string sd(float*,int);

vector<Point> d2_sample(vector<Point> &, vector<Point> &, vector<int> &, int, int);
Point mean_heuristic(vector<Point> &);
vector<Point> independent_sample(vector<Point> &, vector<Point> &, float);
