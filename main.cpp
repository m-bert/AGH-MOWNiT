#include <iostream>
#include <float.h>
#include <math.h>

#define BITS_IN_BYTE CHAR_BIT

void count_bytes(int *float_size_bytes, int *double_size_bytes, int *long_double_size_bytes);
void count_mantissa_bits(int *float_mantissa_bits, int *double_mantissa_bits, int *long_double_mantissa_bits);
// void count_exponent_bits(int *float_exponent_bits, int *double_exponent_bits, int *long_double_exponent_bits);

int main(int argc, char *argv[])
{
    int float_size_bytes, double_size_bytes, long_double_size_bytes;
    int float_mantissa_bits, double_mantissa_bits, long_double_mantissa_bits;

    count_bytes(&float_size_bytes, &double_size_bytes, &long_double_size_bytes);
    count_mantissa_bits(&float_mantissa_bits, &double_mantissa_bits, &long_double_mantissa_bits);

    // Count exponent bits

    int float_exponent_bits = float_size_bytes * BITS_IN_BYTE - float_mantissa_bits;
    int double_exponent_bits = double_size_bytes * BITS_IN_BYTE - double_mantissa_bits;
    int long_double_exponent_bits = long_double_size_bytes * BITS_IN_BYTE - long_double_mantissa_bits;

    std::cout << "====================================" << std::endl;
    std::cout << "Exponent bits:" << std::endl;
    std::cout << "float: " << float_exponent_bits << std::endl;
    std::cout << "double: " << double_exponent_bits << std::endl;
    std::cout << "long double: " << long_double_exponent_bits << std::endl;
    std::cout << "====================================" << std::endl;
    std::cout << std::endl;

    return 0;
}

void count_bytes(int *float_size_bytes, int *double_size_bytes, int *long_double_size_bytes)
{
    // Computing size in bytes
    *float_size_bytes = sizeof(float);
    *double_size_bytes = sizeof(double);
    *long_double_size_bytes = sizeof(long double);

    std::cout << "====================================" << std::endl;
    std::cout << "Sizes in bytes:" << std::endl;
    std::cout << "float: " << *float_size_bytes << std::endl;
    std::cout << "double: " << *double_size_bytes << std::endl;
    std::cout << "long double: " << *long_double_size_bytes << std::endl;
    std::cout << "====================================" << std::endl;
    std::cout << std::endl;
}

void count_mantissa_bits(int *float_mantissa_bits, int *double_mantissa_bits, int *long_double_mantissa_bits)
{
    float f = 1.0f;
    for (*float_mantissa_bits = 0; 1.0f + f != 1.0f; (*float_mantissa_bits)++)
    {
        f /= 2;
    }

    double d = 1.0;
    for (*double_mantissa_bits = 0; 1.0 + d != 1.0; (*double_mantissa_bits)++)
    {
        d /= 2;
    }

    long double ld = 1.0l;
    for (*long_double_mantissa_bits = 0; 1.0l + ld != 1.0l; (*long_double_mantissa_bits)++)
    {
        ld /= 2;
    }

    std::cout << "====================================" << std::endl;
    std::cout << "Mantissa size in bits:" << std::endl;
    std::cout << "float: " << *float_mantissa_bits << std::endl;
    std::cout << "double: " << *double_mantissa_bits << std::endl;
    std::cout << "long double: " << *long_double_mantissa_bits << std::endl;
    std::cout << "====================================" << std::endl;
    std::cout << std::endl;
}

// void count_exponent_bits(int *float_exponent_bits, int *double_exponent_bits, int *long_double_exponent_bits)
// {
//     float_exponent_bits =
// }