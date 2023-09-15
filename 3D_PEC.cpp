//
// Created by Ivan Okhmatovskii on 1/23/23.
//

#include <iostream>
#include <math.h>
#include <vector>

using namespace std;


double* g(const double *t, const double t0, const double tau)
{
    auto gaussian = new double[sizeof(t)];
    for (int i = 0; i < sizeof(t);++i)
    {
        gaussian[i] = exp(-(t[i]-t0)*(t[i]-t0) / tau/tau);
    }
    return gaussian;
}


template<typename T>
std::vector<double> linspace(T start_in, T end_in, int num_in)
{

    std::vector<double> linspaced;

    double start = static_cast<double>(start_in);
    double end = static_cast<double>(end_in);
    double num = static_cast<double>(num_in);

    if (num == 0) { return linspaced; }
    if (num == 1)
    {
        linspaced.push_back(start);
        return linspaced;
    }

    double delta = (end - start) / (num - 1);

    for(int i=0; i < num-1; ++i)
    {
        linspaced.push_back(start + delta * i);
    }
    linspaced.push_back(end); // I want to ensure that start and end
    // are exactly the same as the input
    return linspaced;
}

int main()
{
    cout << "hello world" << endl;

    std::vector<double> T = linspace(0, 10, 9);

    std::cout << T[5] << std::endl;

    return 0;
}