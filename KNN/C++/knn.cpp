#include <iostream>
#include <string>
#include <cstdlib>
#include <cmath>

#define Count 2

using namespace std;

// a sample like feature = {"action", {1, 99}, 0.0}
struct KNN {
    string feature;
    float count[Count];
    float distance;
    // KNN(const string f, int* c, const int Count) {
        // feature = f;
        // for (size_t i = 0; i < Count; i++) {
            // count[i] = a[i];
        // }
    // }
};

void calculateDistance(struct KNN *knn, const int N, struct KNN *sample) {
    for (size_t i = 0; i < N; i++) {
        float distance = 0.0;
        for (size_t j = 0; j < Count; j++) {
            distance += pow(knn[i].count[j] - sample->count[j], 2);
        }
        distance = sqrt(distance);
        knn[i].distance = distance;
    }
}

void sortKNN(struct KNN *knn, const int N) {
    for (size_t i = 0; i < N; i++) {
        for (size_t j = i + 1; j < N; j++) {
            if (knn[i].distance > knn[j].distance) {
                float t = knn[i].distance;
                knn[i].distance = knn[j].distance;
                knn[j].distance = t;
            }
        }
    }
}

int sum(int *a, const int N) {
    int t = 0;
    for (size_t i = 0; i < N; i++) {
        t += a[i];
    }
    return t;
}

string similiar(const int k, struct KNN* knn, const int N, struct KNN *sample) {
    if (k > N) {
        cout << "k cannot be bigger than N" << endl;
        return "error";
    }
    // calculate distance
    calculateDistance(knn, N, sample);
    // sort knn
    sortKNN(knn, N);
    
    int *label = new int[k];
    for (size_t i = 0; i < k; i++) {
        label[i] = 0;
    }
    
    for (size_t i = 0; i < k; i++) {
        for (size_t j = sum(label, k); j < k; j++) {
            if (knn[i].feature == knn[j].feature) {
                label[i] += 1;
            }
        }
    }
    
    int max = 0; 
    for (size_t i = 1; i < k; i++) {
        if (label[max] < label[i]) max = i;
    }
    
    delete[] label;
    
    return knn[max].feature;
}

int main(int argc, char** argv) {
    KNN x0 = {"Romance", {3, 104}, 0};
    KNN x1 = {"Romance", {2, 100}, 0};
    KNN x2 = {"Romance", {1, 81}, 0};
    KNN x3 = {"Action", {101, 10}, 0};
    KNN x4 = {"Action", {99, 5}, 0};
    KNN x5 = {"Action", {98, 2}, 0};
    KNN x6 = {"Action", {93, 4}, 0};
    
    const int N = 7;
    KNN pl[N] = {x0, x1, x2, x3, x4, x5, x6};
    KNN sample = {"Unknown", {34, 56}, 0};
    
    cout << "this is a " << similiar(5, pl, N, &sample) << " movie when K is 5" << endl;
    cout << "this is a " << similiar(6, pl, N, &sample) << " movie when K is 6" << endl;
    cout << "this is a " << similiar(7, pl, N, &sample) << " movie when K is 7" << endl;
    
    return 0;
}