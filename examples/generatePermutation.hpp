void ND1D(int, int, int, int, int, int, int, int, int, int, int*);
void ND2D(int, int, int, int, int, int, int, int, int, int, int, int*);
void ND3D(int, int, int, int, int, int, int, int, int, int, int, int*);

int* generatePermutation(int argc, char * argv[]) {
  int dim;
  int nx, ny, nz;
  int *order;

  dim=atoi(argv[1]);
  nx=atoi(argv[2])+1;

  switch(dim) {
    case 1:
      ny=1;
      nz=1;
      break;
    case 2:
      ny=nx;
      nz=1;
      break;
    case 3:
      ny=nx;
      nz=nx;
      break;
  }

  if(nx<1 || ny<1 || nz<1) {
    printf("Wrong arguments: nx, ny, [nz] must be positive!\n");
    return NULL;
  }

  if(ny==1 && nz==1) {
    printf("1D Nested Dissection with nx=%d\n",nx);
    if(!(order=(int *) malloc(sizeof(int)*nx))) {
      printf("Could not allocate memory\n");
      return NULL;
    }
    ND1D(1,nx,1,1,1,1,nx,1,1,0,order);
  } else if(nz==1) {
    printf("2D Nested Dissection with nx=%d, ny=%d\n",nx,ny);
    if(!(order=(int *) malloc(sizeof(int)*nx*ny))) {
      printf("Could not allocate memory\n");
      return NULL;
    }
    ND2D(1,nx,1,ny,1,1,nx,ny,1,0,0,order);
  } else {
    printf("3D Nested Dissection with nx=%d, ny=%d, nz=%d\n",nx,ny,nz);
    if(!(order=(int *) malloc(sizeof(int)*nx*ny*nz))) {
      printf("Could not allocate memory\n");
      return NULL;
    }
    ND3D(1,nx,1,ny,1,nz,nx,ny,nz,0,0,order);
  }

  return order;

}

void ND1D(int bx, int ex, int by, int ey, int bz, int ez, int nx, int ny, int nz, int offset, int* order) {
/* 1D Nested Dissection of a line of an nx x ny x nz cuboid mesh                  */
/* Ordering is written in array order with starting positing "offset".            */
  int sx, sy, sz, hx, hy, hz;

  sx=ex-bx+1;
  sy=ey-by+1;
  sz=ez-bz+1;

  if(sx<=0 || sy<=0 || sz <=0)
    return;

  if(sx==1 && sy==1 && sz==1) {
    order[offset]=(bx-1)*ny*nz+(by-1)*nz+bz;
    return;
  }

  if(sy==1 && sz==1) {
    hx=bx+(ex-bx+1)/2;
    ND1D(bx  ,hx-1,by,ey,bz,ez,nx,ny,nz,offset        ,order);
    ND1D(hx+1,ex  ,by,ey,bz,ez,nx,ny,nz,offset+(hx-bx),order);
    order[offset+ex-bx]=(hx-1)*ny*nz+(by-1)*nz+bz;
  } else if (sx==1 && sz==1) {
    hy=by+(ey-by+1)/2;
    ND1D(bx,ex,by  ,hy-1,bz,ez,nx,ny,nz,offset        ,order);
    ND1D(bx,ex,hy+1,ey  ,bz,ez,nx,ny,nz,offset+(hy-by),order);
    order[offset+ey-by]=(bx-1)*ny*nz+(hy-1)*nz+bz;
  } else if (sx==1 && sy==1) {
    hz=bz+(ez-bz+1)/2;
    ND1D(bx,ex,by,ey,bz  ,hz-1,nx,ny,nz,offset        ,order);
    ND1D(bx,ex,by,ey,hz+1,ez  ,nx,ny,nz,offset+(hz-bz),order);
    order[offset+ez-bz]=(bx-1)*ny*nz+(by-1)*nz+hz;
  } else {
    printf("Internal error ND1D\n");
    exit(1);
  }

  return;

}

void ND2D(int bx, int ex, int by, int ey, int bz, int ez, int nx, int ny, int nz, int offset, int cut, int* order) {
/* 2D Nested Dissection of a rectangle of an nx x ny x nz rectangular mesh       */
/* Cut defines how to cut the domain (e.g., if constant z, 0: x-wise, 1: y-wise) */
/* Ordering is written in array order with starting positing "offset".           */
  int sx, sy, sz, hx, hy, hz;

  sx=ex-bx+1;
  sy=ey-by+1;
  sz=ez-bz+1;

  if(sx>1 && sy>1 && sz>1) {
    printf("Internal error ND2D\n");
    exit(1);
  }

  if(sx<=0 || sy<=0 || sz<=0) return;

  if(sx==2 && sy==2) {
    order[offset]  =(bx-1)*nx*nz+(by-1)*nz+bz;
    order[offset+1]=(bx)  *nx*nz+(by)  *nz+bz;
    order[offset+2]=(bx)  *nx*nz+(by-1)*nz+bz;
    order[offset+3]=(bx-1)*nx*nz+(by)  *nz+bz;
    return;
  }

  if(sx==2 && sz==2) {
    order[offset]  =(bx-1)*nx*nz+(by-1)*nz+bz;
    order[offset+1]=(bx)  *nx*nz+(by-1)*nz+bz+1;
    order[offset+2]=(bx)  *nx*nz+(by-1)*nz+bz;
    order[offset+3]=(bx-1)*nx*nz+(by-1)*nz+bz+1;
    return;
  }

  if(sy==2 && sz==2) {
    order[offset]  =(bx-1)*nx*nz+(by-1)*nz+bz;
    order[offset+1]=(bx-1)*nx*nz+(by-1)*nz+bz+1;
    order[offset+2]=(bx-1)*nx*nz+(by)  *nz+bz;
    order[offset+3]=(bx-1)*nx*nz+(by)  *nz+bz+1;
    return;
  }

  if((sx==1 && sy==1) || (sx==1 && sz==1) || (sy==1 && sz==1)) {
    ND1D(bx,ex,by,ey,bz,ez,nx,ny,nz,offset,order);
    return;
  }

  if(sz==1) {
    if(cut==0) {
      hx=bx+(ex-bx+1)/2;
      ND2D(bx  ,hx-1,by,ey,bz,ez,nx,ny,nz,offset           ,1,order);
      ND2D(hx+1,ex  ,by,ey,bz,ez,nx,ny,nz,offset+(hx-bx)*sy,1,order);
      ND1D(hx  ,hx  ,by,ey,bz,ez,nx,ny,nz,offset+(ex-bx)*sy,  order);
    } else {
      hy=by+(ey-by+1)/2;
      ND2D(bx,ex,by  ,hy-1,bz,ez,nx,ny,nz,offset           ,0,order);
      ND2D(bx,ex,hy+1,ey  ,bz,ez,nx,ny,nz,offset+(hy-by)*sx,0,order);
      ND1D(bx,ex,hy  ,hy  ,bz,ez,nx,ny,nz,offset+(ey-by)*sx,  order);
    }
  } else if(sy==1) {
    if(cut==0) {
      hx=bx+(ex-bx+1)/2;
      ND2D(bx  ,hx-1,by,ey,bz,ez,nx,ny,nz,offset           ,1,order);
      ND2D(hx+1,ex  ,by,ey,bz,ez,nx,ny,nz,offset+(hx-bx)*sz,1,order);
      ND1D(hx  ,hx  ,by,ey,bz,ez,nx,ny,nz,offset+(ex-bx)*sz,  order);
    } else {
      hz=bz+(ez-bz+1)/2;
      ND2D(bx,ex,by,ey,bz  ,hz-1,nx,ny,nz,offset           ,0,order);
      ND2D(bx,ex,by,ey,hz+1,ez  ,nx,ny,nz,offset+(hz-bz)*sx,0,order);
      ND1D(bx,ex,by,ey,hz  ,hz  ,nx,ny,nz,offset+(ez-bz)*sx,  order);
    }
  } else if(sx==1) {
    if(cut==0) {
      hy=by+(ey-by+1)/2;
      ND2D(bx,ex,by  ,hy-1,bz,ez,nx,ny,nz,offset           ,0,order);
      ND2D(bx,ex,hy+1,ey  ,bz,ez,nx,ny,nz,offset+(hy-by)*sz,0,order);
      ND1D(bx,ex,hy  ,hy  ,bz,ez,nx,ny,nz,offset+(ey-by)*sz,  order);
    } else {
      hz=bz+(ez-bz+1)/2;
      ND2D(bx,ex,by,ey,bz  ,hz-1,nx,ny,nz,offset           ,0,order);
      ND2D(bx,ex,by,ey,hz+1,ez  ,nx,ny,nz,offset+(hz-bz)*sy,0,order);
      ND1D(bx,ex,by,ey,hz  ,hz  ,nx,ny,nz,offset+(ez-bz)*sy,  order);
    }
  }

  return;
}

void ND3D(int bx, int ex, int by, int ey, int bz, int ez, int nx, int ny, int nz, int offset, int cut, int* order) {
/* 3D Nested Dissection of subdomain [bx:ex]x[by:ey]x[bz:ez] of an nx x ny x nz rectangular mesh */
/* Cut defines how to cut the domain (0: x-wise, 1: y-wise, 2: z-wise)                           */
/* Points of the domain are assumed to be numbered y-wise.                                       */
/* Ordering is written in array order with starting positing "offset".                           */
  int sx, sy, sz, hx, hy, hz;

  sx=ex-bx+1;
  sy=ey-by+1;
  sz=ez-bz+1;

  if(sx<=0 || sy<=0 || sz<=0) return;

  if(sx==2 && sy==2 && sz==2) {
    order[offset]  =(bx-1)*nx*nz+(by-1)*nz+bz;
    order[offset+1]=(bx-1)*nx*nz+(by-1)*nz+bz+1;
    order[offset+2]=(bx-1)*nx*nz+(by)  *nz+bz;
    order[offset+3]=(bx-1)*nx*nz+(by)  *nz+bz+1;;
    order[offset+4]=(bx)  *nx*nz+(by-1)*nz+bz;
    order[offset+5]=(bx)  *nx*nz+(by-1)*nz+bz+1;
    order[offset+6]=(bx)  *nx*nz+(by)  *nz+bz;
    order[offset+7]=(bx)  *nx*nz+(by)  *nz+bz+1;;
    return;
  }

  if((sx==1 && sy==1) || (sx==1 && sz==1) || (sy==1 && sz==1)) {
    ND1D(bx,ex,by,ey,bz,ez,nx,ny,nz,offset,order);
    return;
  }

  if(sx==1 || sy==1 || sz==1) {
    ND2D(bx,ex,by,ey,bz,ez,nx,ny,nz,offset,0,order);
    return;
  }

  if(cut==0) {
    hx=bx+(ex-bx+1)/2;
    ND3D(bx  ,hx-1,by,ey,bz,ez,nx,ny,nz,offset              ,1,order);
    ND3D(hx+1,ex  ,by,ey,bz,ez,nx,ny,nz,offset+(hx-bx)*sy*sz,1,order);
    ND2D(hx  ,hx  ,by,ey,bz,ez,nx,ny,nz,offset+(ex-bx)*sy*sz,0,order);
  } else if(cut==1) {
    hy=by+(ey-by+1)/2;
    ND3D(bx,ex,by  ,hy-1,bz,ez,nx,ny,nz,offset              ,2,order);
    ND3D(bx,ex,hy+1,ey  ,bz,ez,nx,ny,nz,offset+(hy-by)*sx*sz,2,order);
    ND2D(bx,ex,hy  ,hy  ,bz,ez,nx,ny,nz,offset+(ey-by)*sx*sz,0,order);
  } else {
    hz=bz+(ez-bz+1)/2;
    ND3D(bx,ex,by,ey,bz  ,hz-1,nx,ny,nz,offset              ,0,order);
    ND3D(bx,ex,by,ey,hz+1,ez  ,nx,ny,nz,offset+(hz-bz)*sx*sy,0,order);
    ND2D(bx,ex,by,ey,hz  ,hz  ,nx,ny,nz,offset+(ez-bz)*sx*sy,0,order);
  }

  return;
}
