#include <iostream>
#include <float.h>
#include <math.h>

#define BITS_IN_BYTE CHAR_BIT

void count_bytes();
void count_mantissa_bits();
void print_exponent_bits();
void calculate_epsilons();
void check_zeros();
void check_infinities();
void checkNaN();

int main(int argc, char *argv[])
{

    count_bytes();
    count_mantissa_bits();
    print_exponent_bits();
    calculate_epsilons();
    check_zeros();
    check_infinities();
    checkNaN();

    return 0;
}

void count_bytes()
{
    std::cout << "====================================" << std::endl;
    std::cout << "Sizes in bytes:" << std::endl;
    std::cout << "float: " << sizeof(float) << std::endl;
    std::cout << "double: " << sizeof(double) << std::endl;
    std::cout << "long double: " << sizeof(long double) << std::endl;
    std::cout << "====================================" << std::endl;
    std::cout << std::endl;
}

void count_mantissa_bits()
{
    int float_mantissa_bits;
    float f = 1.0f;
    for (float_mantissa_bits = 0; 1.0f + f != 1.0f; ++float_mantissa_bits)
    {
        f /= 2;
    }

    int double_mantissa_bits;
    double d = 1.0;
    for (double_mantissa_bits = 0; 1.0 + d != 1.0; ++double_mantissa_bits)
    {
        d /= 2;
    }

    int long_double_mantissa_bits;
    long double ld = 1.0l;
    for (long_double_mantissa_bits = 0; 1.0l + ld != 1.0l; ++long_double_mantissa_bits)
    {
        ld /= 2;
    }

    std::cout << "====================================" << std::endl;
    std::cout << "Mantissa size in bits:" << std::endl;
    std::cout << "float: " << float_mantissa_bits << "(" << FLT_MANT_DIG << ")" << std::endl;
    std::cout << "double: " << double_mantissa_bits << "(" << DBL_MANT_DIG << ")" << std::endl;
    std::cout << "long double: " << long_double_mantissa_bits << "(" << LDBL_MANT_DIG << ")" << std::endl;
    std::cout << "====================================" << std::endl;
    std::cout << std::endl;
}

void print_exponent_bits()
{
    int float_exponent_bits = std::ceil(std::log2(FLT_MAX_EXP - FLT_MIN_EXP + 1));
    int double_exponent_bits = std::ceil(std::log2(DBL_MAX_EXP - DBL_MIN_EXP + 1));
    int long_double_exponent_bits = std::ceil(std::log2(LDBL_MAX_EXP - LDBL_MIN_EXP + 1));

    std::cout << "====================================" << std::endl;
    std::cout << "Exponent bits:" << std::endl;
    std::cout << "float: " << float_exponent_bits << std::endl;
    std::cout << "double: " << double_exponent_bits << std::endl;
    std::cout << "long double: " << long_double_exponent_bits << std::endl;
    std::cout << "====================================" << std::endl;
    std::cout << std::endl;
}

void calculate_epsilons()
{
    float float_epsilon = 1.0f, float_tmp_epsilon = 1.0f;
    while (1.0f + float_tmp_epsilon != 1.0f)
    {
        float_epsilon = float_tmp_epsilon;
        float_tmp_epsilon /= 2.0f;
    }

    double double_epsilon = 1.0, double_tmp_epsilon = 1.0;
    while (1.0 + double_tmp_epsilon != 1.0)
    {
        double_epsilon = double_tmp_epsilon;
        double_tmp_epsilon /= 2.0;
    }

    long double long_double_epsilon = 1.0l, long_double_tmp_epsilon = 1.0l;
    while (1.0l + long_double_tmp_epsilon != 1.0l)
    {
        long_double_epsilon = long_double_tmp_epsilon;
        long_double_tmp_epsilon /= 2.0l;
    }

    std::cout << "====================================" << std::endl;
    std::cout << "Machine epsilon:" << std::endl;
    std::cout << "float: " << float_epsilon << std::endl;
    std::cout << "double: " << double_epsilon << std::endl;
    std::cout << "long double: " << long_double_epsilon << std::endl;
    std::cout << "====================================" << std::endl;
    std::cout << std::endl;
}

void check_zeros()
{
    float positive_zero = 1.0f / 10000000000000000000000000000000000000000000000000.0;
    float negative_zero = -1.0f / 10000000000000000000000000000000000000000000000000.0;

    std::cout << "====================================" << std::endl;
    std::cout << "Zero check:" << std::endl;
    std::cout << "Positive: " << positive_zero << "  | Sign bit: " << signbit(positive_zero) << std::endl;
    std::cout << "Negative: " << negative_zero << " | Sign bit: " << signbit(negative_zero) << std::endl;
    std::cout << "-0 == 0 -> " << (positive_zero == negative_zero) << std::endl;
    std::cout << "====================================" << std::endl;
    std::cout << std::endl;
}

void check_infinities()
{
    std::cout << "====================================" << std::endl;
    std::cout << "Infinity check:" << std::endl;
    std::cout << "Positive infinity: " << (1 / 0.0) << std::endl;
    std::cout << "Negative infinity: " << (-1 / 0.0) << std::endl;
    std::cout << "-inf < FLT_MIN -> " << ((-1 / 0.0) < FLT_MIN) << std::endl;
    std::cout << "====================================" << std::endl;
    std::cout << std::endl;
}

void checkNaN()
{
    float a = sqrt(-2);

    std::cout << "====================================" << std::endl;
    std::cout << "NaN check:" << std::endl;
    std::cout << "Built-in NaN: " << NAN << std::endl;
    std::cout << "a = sqrt(-2) = " << a << std::endl;
    std::cout << "a == a -> " << (a == a) << std::endl;
    std::cout << "isnan(a) -> " << isnan(a) << std::endl;
    std::cout << "====================================" << std::endl;
    std::cout << std::endl;
}