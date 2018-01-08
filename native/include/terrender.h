/* From https://youtu.be/zmtHaZG7pPc?t=22m19s */

struct terrender_error {
    char *message;
    int failed;
    int code;
};

void terrender_init(void);

unsigned long terrender_order_overlapping_triangles(
	const double *faces, unsigned long nfaces,
	unsigned long *output, unsigned long output_size,
	struct terrender_error *error);
