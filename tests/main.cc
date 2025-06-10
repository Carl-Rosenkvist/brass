// tests/main.cpp
#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"   // path relative to your tests folder

int main(int argc, char** argv) {
    doctest::Context context;

    context.applyCommandLine(argc, argv);

    // custom setup...
    int res = context.run(); // run all registered tests

    // important - query flags (and --exit)
    if(context.shouldExit()) 
        return res;

    // your app code here, if you want to run tests + app
    return res;
}
