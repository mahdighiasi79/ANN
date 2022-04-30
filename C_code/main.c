#include <stdio.h>
#include <stdlib.h>

int main() {
    FILE *fp = fopen("../t10k-labels.idx1-ubyte", "rb");
    unsigned char *buffer = (unsigned char *) malloc(4);
    fseek(fp, 0, SEEK_SET);
    fread(buffer, 1, 4, fp);
    for (int i = 0; i < 4; i++)
        printf("%u\n", buffer[i]);
    return 0;
}
