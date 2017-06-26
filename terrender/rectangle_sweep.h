#ifdef __cplusplus
extern "C" {
#endif

struct rect {
	double x1, x2, y1, y2;
	long idx;
};

void * rectangle_sweep_init(void);
long rectangle_sweep_push(void *, double, double, double, double, long, long *, long *);
void rectangle_sweep_free(void *);

#ifdef __cplusplus
}
#endif
