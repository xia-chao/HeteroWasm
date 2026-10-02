void vector_add(const int *a, const int *b, int *c, unsigned long n) {
    for (unsigned long i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

void saxpy(int alpha, const int *a, const int *b, int *c, unsigned long n) {
    for (unsigned long i = 0; i < n; i++) {
        c[i] = alpha * a[i] + b[i];
    }
}

void elementwise_mul(const int *a, const int *b, int *c, unsigned long n) {
    for (unsigned long i = 0; i < n; i++) {
        c[i] = a[i] * b[i];
    }
}
