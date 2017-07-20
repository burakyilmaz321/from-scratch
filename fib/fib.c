#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) 
{
    if (argc == 2)
    {
        if (atoi(argv[1]) == 0)
        {
            printf("Pass a non-zero integer!\n");
        }
        
        int a, b, c;

        a = 0;
        b = 1;

        do 
        {
            printf("%d\n", a);
            c = a + b;
            a = b;
            b = c;
        } while (a < atoi(argv[1]));
    }
    else if (argc > 2)
    {
        printf("Too many argument given!\n");
    }
    else
    {
        printf("Give one argument!\n");
    }
}