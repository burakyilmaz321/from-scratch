#include <stdio.h>

int main(int argc, char *argv[]) 
{
    if (argc == 2)
    {
        int a, b, c;

        a = 0;
        b = 1;

        do 
        {
            printf("%d\n", a);
            c = a + b;
            a = b;
            b = c;
        } while (a < 255);
    }
    else if (argc > 2)
    {
        printf("Too many argument given!\n");
    }
    else
    {
        printf("Only one argument expected!\n");
    }
}