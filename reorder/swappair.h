#ifndef SWAPPAIR_H
#define SWAPPAIR_H
#include <functional>

class SwapPair {
public:
    SwapPair(int _v1, int _v2, int lg, int rg) : v1(_v1), v2(_v2), 
    left_gain(lg), right_gain(rg){}

    void print()
    {
       std::cout << "swap (" + std::to_string(v1) + ", " + std::to_string(v2) + "): swap_gain = (" + std::to_string(left_gain) + ", " + std::to_string(right_gain) + ")\n";
    }

public:
    int v1;
    int v2;
    int left_gain;
    int right_gain;
};

class SwapPairs {
public:
    SwapPairs(int _v1, int _v2, int lg, int rg) : v1(_v1), v2(_v2), 
    left_gain(lg), right_gain(rg), v3(-1), v4(-1){}

    SwapPairs(int code, int lseg, int rseg, int lg, int rg) : left_seg(), right_seg(), left_gain(lg), right_gain(rg) {
        v1 = lseg*4 + pairtab[code][0];
        v2 = rseg*4 + (pairtab[code][1]-4);
        v3 = lseg*4 + pairtab[code][2];
        v4 = rseg*4 + (pairtab[code][3]-4);
    }

    void print()
    {
        if (v3 != -1 && v4 != -1)
            std::cout << "swap (" + std::to_string(v1) + ", " + std::to_string(v2) + ") and (" + std::to_string(v3) + ", " + std::to_string(v4) + "): swap_gain = (" + std::to_string(left_gain) + ", " + std::to_string(right_gain) + ")\n";
        else
            std::cout << "swap (" + std::to_string(v1) + ", " + std::to_string(v2) + ")            : swap_gain = (" + std::to_string(left_gain) + ", " + std::to_string(right_gain) + ")\n";
    }

public:
    int v1;
    int v2;
    int v3;
    int v4;
    int left_gain;
    int right_gain;

    int left_seg;
    int right_seg;

private:
    std::vector<std::vector<int>> pairtab 
    { 
        {0, 6, 3, 7}, 
        {0, 5, 3, 7}, 
        {0, 5, 3, 6},
        {0, 4, 3, 7}, 
        {0, 4, 3, 6}, 
        {0, 4, 3, 5},

        {2, 4, 3, 5},
        {2, 4, 3, 6},
        {2, 4, 3, 7},
        {2, 5, 3, 6},
        {2, 5, 3, 7},
        {2, 6, 3, 7},

        {1, 4, 3, 5},
        {1, 4, 3, 6},
        {1, 4, 3, 7},
        {1, 5, 3, 6},
        {1, 5, 3, 7},
        {1, 6, 3, 7},

        {1, 4, 2, 5},
        {1, 4, 2, 6},
        {1, 4, 2, 7},
        {1, 5, 2, 6},
        {1, 5, 2, 7},
        {1, 6, 2, 7},

        {0, 6, 2, 7},
        {0, 5, 2, 7},
        {0, 5, 2, 6},
        {0, 4, 2, 7},
        {0, 4, 2, 6},
        {0, 4, 2, 5},

        {0, 4, 1, 5},
        {0, 4, 1, 6},
        {0, 4, 1, 7},
        {0, 5, 1, 7},
        {0, 5, 1, 6},
        {0, 6, 1, 5}
    }; 
};

// for dequeue
bool funComparator(SwapPair *p1, SwapPair *p2)
{
    if (p1->left_gain+p1->right_gain == p2->left_gain+p2->right_gain) 
    {
        return p1->right_gain >= p2->right_gain;
    }

    return  p1->left_gain+p1->right_gain > p2->left_gain+p2->right_gain;
}

// for priority queue
bool Comparator(SwapPair *p1, SwapPair *p2)
{
    if (p1->left_gain+p1->right_gain == p2->left_gain+p2->right_gain) 
    {
        return p1->right_gain <= p2->right_gain;
    }

    return  p1->left_gain+p1->right_gain < p2->left_gain+p2->right_gain;
}

bool Comparators(SwapPairs *p1, SwapPairs *p2)
{
    if (p1->left_gain+p1->right_gain == p2->left_gain+p2->right_gain) 
    {
        return p1->right_gain <= p2->right_gain;
    }

    return  p1->left_gain+p1->right_gain < p2->left_gain+p2->right_gain;
}

void showpq(std::priority_queue<SwapPair*, std::vector<SwapPair*>, 
            std::function<bool(SwapPair*, SwapPair*)>> gq)
{
    std::priority_queue<SwapPair*, std::vector<SwapPair*>, std::function<bool(SwapPair*, SwapPair*)>> g = gq;
    while (!g.empty()) {
        (g.top())->print();
        g.pop();
    }
    std::cout << '\n';
}

void showpq(std::priority_queue<SwapPairs*, std::vector<SwapPairs*>, 
            std::function<bool(SwapPairs*, SwapPairs*)>> gq)
{
    std::priority_queue<SwapPairs*, std::vector<SwapPairs*>, std::function<bool(SwapPairs*, SwapPairs*)>> g = gq;
    while (!g.empty()) {
        (g.top())->print();
        g.pop();
    }
    std::cout << '\n';
}

void showpq(std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, 
            std::function<bool(std::pair<int, int>&, std::pair<int, int>&)>> gq)
{
    std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, std::function<bool(std::pair<int, int>&, std::pair<int, int>&)>> g = gq;
    while (!g.empty()) {
        std::cout << (g.top()).first << "(" << (g.top()).second << "), ";
        g.pop();
    }
    std::cout << '\n';
}



#endif