/*--------------------------------------------------------------------
 * c     eta-direction flux differences             
 * c-------------------------------------------------------------------*/
void foo() i{
#pragma omp for
  for (i = 1; i < grid_points[0]-1; i++) {
    xi = (double)i * dnxm1;
    
    for (k = 1; k < grid_points[2]-1; k++) {
      zeta = (double)k * dnzm1;

      for (j = 0; j < grid_points[1]; j++) {
  eta = (double)j * dnym1;

  exact_solution(xi, eta, zeta, dtemp);
  for (m = 0; m < 5; m++) {
    ue[j][m] = dtemp[m];
  }
                  
  dtpp = 1.0/dtemp[0];

  for (m = 1; m <= 4; m++) {
    buf[j][m] = dtpp * dtemp[m];
  }

  cuf[j]   = buf[j][2] * buf[j][2];
  buf[j][0] = cuf[j] + buf[j][1] * buf[j][1] + 
    buf[j][3] * buf[j][3];
  q[j] = 0.5*(buf[j][1]*ue[j][1] + buf[j][2]*ue[j][2] +
        buf[j][3]*ue[j][3]);
      }

      for (j = 1; j < grid_points[1]-1; j++) {
  jm1 = j-1;
  jp1 = j+1;

  x = calc[i]; y = calc[j];
                  
  forcing[i][j][k][0] = forcing[i][j][k][0] -
    ty2*( ue[jp1][2]-ue[jm1][2] )+
    dy1ty1*(ue[jp1][0]-2.0*ue[j][0]+ue[jm1][0]);

  forcing[i][j][k][1] = forcing[i][j][k][1] -
    ty2*(ue[jp1][1]*buf[jp1][2]-ue[jm1][1]*buf[jm1][2])+
    yycon2*(buf[jp1][1]-2.0*buf[j][1]+buf[jm1][1])+
    dy2ty1*( ue[jp1][1]-2.0* ue[j][1]+ ue[jm1][1]);

  forcing[i][j][k][2] = forcing[i][j][k][2] -
    ty2*((ue[jp1][2]*buf[jp1][2]+c2*(ue[jp1][4]-q[jp1]))-
         (ue[jm1][2]*buf[jm1][2]+c2*(ue[jm1][4]-q[jm1])))+
    yycon1*(buf[jp1][2]-2.0*buf[j][2]+buf[jm1][2])+
    dy3ty1*( ue[jp1][2]-2.0*ue[j][2] +ue[jm1][2]);

  forcing[i][j][k][3] = forcing[i][j][k][3] -
    ty2*(ue[jp1][3]*buf[jp1][2]-ue[jm1][3]*buf[jm1][2])+
    yycon2*(buf[jp1][3]-2.0*buf[j][3]+buf[jm1][3])+
    dy4ty1*( ue[jp1][3]-2.0*ue[j][3]+ ue[jm1][3]);

  forcing[i][j][k][4] = forcing[i][j][k][4] -
    ty2*(buf[jp1][2]*(c1*ue[jp1][4]-c2*q[jp1])-
         buf[jm1][2]*(c1*ue[jm1][4]-c2*q[jm1]))+
    0.5*yycon3*(buf[jp1][0]-2.0*buf[j][0]+
                      buf[jm1][0])+
    yycon4*(cuf[jp1]-2.0*cuf[j]+cuf[jm1])+
    yycon5*(buf[jp1][4]-2.0*buf[j][4]+buf[jm1][4])+
    dy5ty1*(ue[jp1][4]-2.0*ue[j][4]+ue[jm1][4]);
      }


}
}
}
