
__attribute__((device)) void comp(double *x) {
  double y=_FPC_CHECK_D_(0.0, 3, "prog_1.cu"), z;
  x[0] = _FPC_CHECK_D_(y*z, 4, "prog_1.cu");


}
