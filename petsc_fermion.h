#ifndef _PETSC_FERMION_H
#define _PETSC_FERMION_H

/****************************************************************************************************************
 * Need something better for the mass parameter and other action information.
 ****************************************************************************************************************
 */
typedef struct {
  PetscReal M;
} WilsonParameters;
typedef struct {
  PetscInt Ls;
  PetscReal M5;
  PetscReal m;
} DomainWallParameters;

typedef struct {
  PetscInt Ls;
  PetscReal c;
  PetscReal b;
  PetscReal M5;
  PetscReal m;
} MobiusDomainWallParameters;

// Must provide these in a single unit of compilation
extern WilsonParameters             TheWilsonParameters;
extern DomainWallParameters         TheDomainWallParameters;
extern MobiusDomainWallParameters   TheMobiusDomainWallParameters;

// Old parameters
//const PetscReal M = 1.0; // M5
//const PetscReal m = 0.01; // mf
//const int Ls=8;

static WilsonParameters *GetWilsonParameters(void)
{
  return &TheWilsonParameters;
}
static void SetWilsonParameters(WilsonParameters *_p)
{
  TheWilsonParameters = *_p;
}
static DomainWallParameters *GetDomainWallParameters(void)
{
  return &TheDomainWallParameters;
}
static void SetDomainWallParameters(DomainWallParameters *_p)
{
  TheDomainWallParameters = *_p;
}
static MobiusDomainWallParameters *GetMobiusDomainWallParameters(void)
{
  return &TheMobiusDomainWallParameters;
}
static void SetMobiusDomainWallParameters(MobiusDomainWallParameters *_p)
{
  TheMobiusDomainWallParameters = *_p;
}
/*****************************************************************************************************************
 * Four spin fermion utils
 *****************************************************************************************************************
 */
static PetscErrorCode GammaMuTimesSpinor(PetscInt mu, PetscInt ldx, PetscScalar f[])
{
  const PetscScalar fin[4] = {f[0 * ldx], f[1 * ldx], f[2 * ldx], f[3 * ldx]};

  PetscFunctionBeginHot;
  switch (mu) {
  case 0:
    f[0 * ldx] = PETSC_i * fin[3];
    f[1 * ldx] = PETSC_i * fin[2];
    f[2 * ldx] = -PETSC_i * fin[1];
    f[3 * ldx] = -PETSC_i * fin[0];
    break;
  case 1:
    f[0 * ldx] = -fin[3];
    f[1 * ldx] = fin[2];
    f[2 * ldx] = fin[1];
    f[3 * ldx] = -fin[0];
    break;
  case 2:
    f[0 * ldx] = PETSC_i * fin[2];
    f[1 * ldx] = -PETSC_i * fin[3];
    f[2 * ldx] = -PETSC_i * fin[0];
    f[3 * ldx] = PETSC_i * fin[1];
    break;
  case 3:
    f[0 * ldx] = fin[2];
    f[1 * ldx] = fin[3];
    f[2 * ldx] = fin[0];
    f[3 * ldx] = fin[1];
    break;
  case 4: /*Gamma 5*/
    f[0 * ldx] = fin[0];
    f[1 * ldx] = fin[1];
    f[2 * ldx] =-fin[2];
    f[3 * ldx] =-fin[3];
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Direction for gamma %" PetscInt_FMT " not in [0, 5)", mu);
  }
  PetscFunctionReturn(0);
}
// Apply (1 \pm \gamma_\mu)/2
static PetscErrorCode SpinProject(PetscInt mu, PetscBool minus, PetscInt ldx, PetscScalar f[])
{
  const PetscReal   sign   = (minus )   ? -1. : 1.;
  const PetscScalar fin[4] = {f[0 * ldx], f[1 * ldx], f[2 * ldx], f[3 * ldx]};

  PetscFunctionBeginHot;
  switch (mu) {
  case 0:
    f[0 * ldx] += sign * PETSC_i * fin[3];
    f[1 * ldx] += sign * PETSC_i * fin[2];
    f[2 * ldx] += sign * -PETSC_i * fin[1];
    f[3 * ldx] += sign * -PETSC_i * fin[0];
    break;
  case 1:
    f[0 * ldx] += -sign * fin[3];
    f[1 * ldx] += sign * fin[2];
    f[2 * ldx] += sign * fin[1];
    f[3 * ldx] += -sign * fin[0];
    break;
  case 2:
    f[0 * ldx] += sign * PETSC_i * fin[2];
    f[1 * ldx] += sign * -PETSC_i * fin[3];
    f[2 * ldx] += sign * -PETSC_i * fin[0];
    f[3 * ldx] += sign * PETSC_i * fin[1];
    break;
  case 3:
    f[0 * ldx] += sign * fin[2];
    f[1 * ldx] += sign * fin[3];
    f[2 * ldx] += sign * fin[0];
    f[3 * ldx] += sign * fin[1];
    break;
  case 4:
    f[0 * ldx] += sign * fin[0];
    f[1 * ldx] += sign * fin[1];
    f[2 * ldx] -= sign * fin[2];
    f[3 * ldx] -= sign * fin[3];
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Direction for gamma %" PetscInt_FMT " not in [0, 5)", mu);
  }
  f[0 * ldx] *= 0.5;
  f[1 * ldx] *= 0.5;
  f[2 * ldx] *= 0.5;
  f[3 * ldx] *= 0.5;
  PetscFunctionReturn(0);
}
/*****************************************************************************************************************
 * Wilson 4d operator
 *****************************************************************************************************************
 */
static PetscErrorCode DwilsonDhop(PetscInt mu,
				  PetscBool forward,
				  PetscBool dag,
				  const PetscScalar U[], const PetscScalar psi[], PetscScalar f[])
{
  PetscScalar tmp[12];

  PetscFunctionBeginHot;
  // Apply U
  for (PetscInt beta = 0; beta < 4; ++beta) {
    if (forward) DMPlex_Mult3D_Internal(U, 1, &psi[beta * 3], &tmp[beta * 3]);
    else DMPlex_MultTranspose3D_Internal(U, 1, &psi[beta * 3], &tmp[beta * 3]);
  }
  // Apply (1 \pm \gamma_\mu)/2 to each color
  PetscBool gforward;
  if ( dag ) {
    if ( forward ) gforward = PETSC_FALSE;
    else           gforward = PETSC_TRUE;
  } else {
    gforward = forward;
  }
  for (PetscInt c = 0; c < 3; ++c) PetscCall(SpinProject(mu, gforward, 3, &tmp[c]));
  // Note that we are subtracting this contribution
  for (PetscInt i = 0; i < 12; ++i) f[i] -= tmp[i];
  PetscFunctionReturn(0);
}
/*
  The assembly loop runs over vertices. Each vertex has 2d edges in its support. The edges are ordered first by the dimension along which they run, and second from smaller to larger index, expect for edges which loop periodically. The vertices on edges are also ordered from smaller to larger index except for periodic edges.
*/
static PetscErrorCode DwilsonInternal(DM dm, Vec u, Vec f, PetscBool dag)
{
  DM                 dmAux;
  Vec                gauge;
  PetscSection       s, sGauge;
  const PetscScalar *ua;
  PetscScalar       *fa, *link;
  PetscInt           dim, vStart, vEnd;

  WilsonParameters *p = GetWilsonParameters();

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetLocalSection(dm, &s));
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  PetscCall(VecGetArrayRead(u, &ua));
  PetscCall(VecGetArray(f, &fa));

  PetscCall(DMGetAuxiliaryVec(dm, NULL, 0, 0, &gauge));
  PetscCall(VecGetDM(gauge, &dmAux));
  PetscCall(DMGetLocalSection(dmAux, &sGauge));
  PetscCall(VecGetArray(gauge, &link));
  // Loop over y
  for (PetscInt v = vStart; v < vEnd; ++v) {
    const PetscInt *supp;
    PetscInt        xdof, xoff;

    PetscCall(DMPlexGetSupport(dm, v, &supp));
    PetscCall(PetscSectionGetDof(s, v, &xdof));
    PetscCall(PetscSectionGetOffset(s, v, &xoff));
    // Diagonal
    for (PetscInt i = 0; i < xdof; ++i) fa[xoff + i] = (p->M + 4) * ua[xoff + i];
    // Loop over mu
    for (PetscInt d = 0; d < dim; ++d) {
      const PetscInt *cone;
      PetscInt        yoff, goff;
      // Left action -(1 + \gamma_\mu)/2 \otimes U^\dagger_\mu(y) \delta_{x - \mu,y} \psi(y)
      PetscCall(DMPlexGetCone(dm, supp[2 * d + 0], &cone));
      PetscCall(PetscSectionGetOffset(s, cone[0], &yoff));
      PetscCall(PetscSectionGetOffset(sGauge, supp[2 * d + 0], &goff));
      PetscCall(DwilsonDhop(d, PETSC_FALSE, dag, &link[goff], &ua[yoff], &fa[xoff]));
      // Right edge -(1 - \gamma_\mu)/2 \otimes U_\mu(x) \delta_{x + \mu,y} \psi(y)
      PetscCall(DMPlexGetCone(dm, supp[2 * d + 1], &cone));
      PetscCall(PetscSectionGetOffset(s, cone[1], &yoff));
      PetscCall(PetscSectionGetOffset(sGauge, supp[2 * d + 1], &goff));
      PetscCall(DwilsonDhop(d, PETSC_TRUE, dag, &link[goff], &ua[yoff], &fa[xoff]));
    }
  }
  PetscCall(VecRestoreArray(f, &fa));
  PetscCall(VecRestoreArray(gauge, &link));
  PetscCall(VecRestoreArrayRead(u, &ua));
  PetscFunctionReturn(0);
}
static PetscErrorCode Dwilson(DM dm, Vec u, Vec f)
{
  return DwilsonInternal(dm,u,f,PETSC_FALSE);
}
static PetscErrorCode DwilsonDag(DM dm, Vec u, Vec f)
{
  return DwilsonInternal(dm,u,f,PETSC_TRUE);
}
static PetscErrorCode DwilsonDagDwilson(DM dm, Vec u, Vec f)
{
  Vec tmp;
  PetscCall(DMCreateGlobalVector(dm, &tmp));
  DwilsonInternal(dm,u,tmp,PETSC_FALSE);
  DwilsonInternal(dm,tmp,f,PETSC_TRUE);
  PetscCall(VecDestroy(&tmp));
  PetscFunctionReturn(0);
}

/*****************************************************************************************************************
 * Shamir domain wall operator
 *****************************************************************************************************************
 */
// Single leg hopping term
static PetscErrorCode DdwfDhop(PetscInt d, PetscBool forward, PetscBool dag, const PetscScalar U[], const PetscScalar psi[], PetscScalar f[])
{
  PetscScalar tmp[12];

  PetscFunctionBeginHot;
  // Apply U
  for (PetscInt beta = 0; beta < 4; ++beta) {
    if (forward) DMPlex_Mult3D_Internal(U, 1, &psi[beta * 3], &tmp[beta * 3]);
    else DMPlex_MultTranspose3D_Internal(U, 1, &psi[beta * 3], &tmp[beta * 3]);
  }
  int gamma[] = {4,0,1,2,3};
  PetscBool gforward;
  if ( dag ) {
    if ( forward ) gforward = PETSC_FALSE;
    else           gforward = PETSC_TRUE;
  } else {
    gforward = forward;
  }
  // Apply (1 \pm \gamma_\mu)/2 to each color for gamma = xyzt
  //  PetscBool gforward = forward;
  for (PetscInt c = 0; c < 3; ++c) PetscCall(SpinProject(gamma[d], gforward, 3, &tmp[c]));
  // Note that we are subtracting this contribution
  for (PetscInt i = 0; i < 12; ++i) f[i] -= tmp[i];
  PetscFunctionReturn(0);
}

static PetscErrorCode DdwfInternal(DM dm, Vec u, Vec f,PetscBool dag)
{
  DM                 dmAux;
  Vec                gauge;
  PetscSection       s, sGauge;
  const PetscScalar *ua;
  PetscScalar       *fa, *link;
  PetscInt           dim, vStart, vEnd;

  DomainWallParameters *p = GetDomainWallParameters();

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetLocalSection(dm, &s));
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  PetscCall(VecGetArrayRead(u, &ua));
  PetscCall(VecGetArray(f, &fa));

  PetscCall(DMGetAuxiliaryVec(dm, NULL, 0, 0, &gauge));
  PetscCall(VecGetDM(gauge, &dmAux));
  PetscCall(DMGetLocalSection(dmAux, &sGauge));
  PetscCall(VecGetArray(gauge, &link));
  // Loop over y

  for (PetscInt v = vStart; v < vEnd; ++v) {
    const PetscInt *supp;
    PetscInt        xdof, xoff;

    PetscCall(DMPlexGetSupport(dm, v, &supp));
    PetscCall(PetscSectionGetDof(s, v, &xdof));
    PetscCall(PetscSectionGetOffset(s, v, &xoff));
    // Diagonal
    for (PetscInt i = 0; i < xdof; ++i) fa[xoff + i] = (4-p->M5 ) * ua[xoff + i];
    // Loop over mu
    for (PetscInt d = 0; d < dim; ++d) {
      const PetscInt *cone;
      PetscInt        yoff, goff;

      // Left action -(1 + \gamma_\mu)/2 \otimes U^\dagger_\mu(y) \delta_{x - \mu,y} \psi(y)
      PetscCall(DMPlexGetCone(dm, supp[2 * d + 0], &cone));
      PetscCall(PetscSectionGetOffset(s, cone[0], &yoff));
      PetscCall(PetscSectionGetOffset(sGauge, supp[2 * d + 0], &goff));
      PetscCall(DdwfDhop(d, PETSC_FALSE, dag, &link[goff], &ua[yoff], &fa[xoff]));
      // Right edge -(1 - \gamma_\mu)/2 \otimes U_\mu(x) \delta_{x + \mu,y} \psi(y)
      PetscCall(DMPlexGetCone(dm, supp[2 * d + 1], &cone));
      PetscCall(PetscSectionGetOffset(s, cone[1], &yoff));
      PetscCall(PetscSectionGetOffset(sGauge, supp[2 * d + 1], &goff));
      PetscCall(DdwfDhop(d, PETSC_TRUE, dag, &link[goff], &ua[yoff], &fa[xoff]));
    }
  }
  PetscCall(VecRestoreArray(f, &fa));
  PetscCall(VecRestoreArray(gauge, &link));
  PetscCall(VecRestoreArrayRead(u, &ua));
  PetscFunctionReturn(0);
}
static PetscErrorCode Ddwf(DM dm, Vec u, Vec f)
{
  return DdwfInternal(dm,u,f,PETSC_FALSE);
}
static PetscErrorCode DdwfDag(DM dm, Vec u, Vec f)
{
  return DdwfInternal(dm,u,f,PETSC_TRUE);
}
static PetscErrorCode DdwfDagDdwf(DM dm, Vec u, Vec f)
{
  Vec tmp;
  PetscCall(DMCreateGlobalVector(dm, &tmp));
  DdwfInternal(dm,u,tmp,PETSC_FALSE);
  DdwfInternal(dm,tmp,f,PETSC_TRUE);
  PetscCall(VecDestroy(&tmp));
  PetscFunctionReturn(0);
}
/*****************************************************************************************************************
 * Mobius domain wall operator
 *****************************************************************************************************************
 */
// Single leg hopping term
static PetscErrorCode DmobiusDhop(PetscInt d, PetscBool forward, PetscBool dag,
				  const PetscScalar U[], const PetscScalar psi[], PetscScalar f[])
{
  PetscScalar tmp[12];

  PetscFunctionBeginHot;
  // Apply U
  for (PetscInt beta = 0; beta < 4; ++beta) {
    if (forward) DMPlex_Mult3D_Internal(U, 1, &psi[beta * 3], &tmp[beta * 3]);
    else DMPlex_MultTranspose3D_Internal(U, 1, &psi[beta * 3], &tmp[beta * 3]);
  }
  int gamma[] = {4,0,1,2,3};
  PetscBool gforward;
  if ( dag ) {
    if ( forward ) gforward = PETSC_FALSE;
    else           gforward = PETSC_TRUE;
  } else {
    gforward = forward;
  }
  // Apply (1 \pm \gamma_\mu)/2 to each color for gamma = xyzt
  for (PetscInt c = 0; c < 3; ++c) PetscCall(SpinProject(gamma[d], gforward, 3, &tmp[c]));
  // Note that we are subtracting this contribution
  for (PetscInt i = 0; i < 12; ++i) f[i] -= tmp[i];
  PetscFunctionReturn(0);
}
static PetscErrorCode DmobiusDslash5(DM dm, Vec u, Vec f,PetscBool dag)
{
  DM                 dmAux;
  Vec                gauge;
  PetscSection       s, sGauge;
  const PetscScalar *ua;
  PetscScalar       *fa, *link;
  PetscInt           dim, vStart, vEnd;

  MobiusDomainWallParameters *p = GetMobiusDomainWallParameters();
  
  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetLocalSection(dm, &s));
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  PetscCall(VecGetArrayRead(u, &ua));
  PetscCall(VecGetArray(f, &fa));

  PetscCall(DMGetAuxiliaryVec(dm, NULL, 0, 0, &gauge));
  PetscCall(VecGetDM(gauge, &dmAux));
  PetscCall(DMGetLocalSection(dmAux, &sGauge));
  PetscCall(VecGetArray(gauge, &link));
  // Loop over y
  for (PetscInt v = vStart; v < vEnd; ++v) {
    const PetscInt *supp;
    PetscInt        xdof, xoff;
    PetscScalar zero=0.0;
    PetscCall(DMPlexGetSupport(dm, v, &supp));
    PetscCall(PetscSectionGetDof(s, v, &xdof));
    PetscCall(PetscSectionGetOffset(s, v, &xoff));
    // No diagonal
    for (PetscInt i = 0; i < xdof; ++i) fa[xoff + i] = zero;

    {
      int d = 0;
      const PetscInt *cone;
      PetscInt        yoff, goff;
      // Left action -(1 + \gamma_\mu)/2 \otimes U^\dagger_\mu(y) \delta_{x - \mu,y} \psi(y)
      PetscCall(DMPlexGetCone(dm, supp[2 * d + 0], &cone));
      PetscCall(PetscSectionGetOffset(s, cone[0], &yoff));
      PetscCall(PetscSectionGetOffset(sGauge, supp[2 * d + 0], &goff));
      PetscCall(DmobiusDhop(d, PETSC_FALSE, dag, &link[goff], &ua[yoff], &fa[xoff]));
      // Right edge -(1 - \gamma_\mu)/2 \otimes U_\mu(x) \delta_{x + \mu,y} \psi(y)
      PetscCall(DMPlexGetCone(dm, supp[2 * d + 1], &cone));
      PetscCall(PetscSectionGetOffset(s, cone[1], &yoff));
      PetscCall(PetscSectionGetOffset(sGauge, supp[2 * d + 1], &goff));
      PetscCall(DmobiusDhop(d, PETSC_TRUE, dag, &link[goff], &ua[yoff], &fa[xoff]));
    }
  }
  PetscCall(VecRestoreArray(f, &fa));
  PetscCall(VecRestoreArray(gauge, &link));
  PetscCall(VecRestoreArrayRead(u, &ua));
  PetscFunctionReturn(0);
}

static PetscErrorCode DmobiusDslash4(DM dm, Vec u, Vec f,PetscBool dag)
{
  DM                 dmAux;
  Vec                gauge;
  PetscSection       s, sGauge;
  const PetscScalar *ua;
  PetscScalar       *fa, *link;
  PetscInt           dim, vStart, vEnd;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetLocalSection(dm, &s));
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  PetscCall(VecGetArrayRead(u, &ua));
  PetscCall(VecGetArray(f, &fa));

  PetscCall(DMGetAuxiliaryVec(dm, NULL, 0, 0, &gauge));
  PetscCall(VecGetDM(gauge, &dmAux));
  PetscCall(DMGetLocalSection(dmAux, &sGauge));
  PetscCall(VecGetArray(gauge, &link));
  // Loop over y
  MobiusDomainWallParameters *p = GetMobiusDomainWallParameters();
  for (PetscInt v = vStart; v < vEnd; ++v) {
    const PetscInt *supp;
    PetscInt        xdof, xoff;

    PetscCall(DMPlexGetSupport(dm, v, &supp));
    PetscCall(PetscSectionGetDof(s, v, &xdof));
    PetscCall(PetscSectionGetOffset(s, v, &xoff));
    // Diagonal
    for (PetscInt i = 0; i < xdof; ++i) fa[xoff + i] = (4 - p->M5 ) * ua[xoff + i];
    // Loop over mu
    for (PetscInt d = 1; d < dim; ++d) { // Omit the fifth dimension
      const PetscInt *cone;
      PetscInt        yoff, goff;

      // Left action -(1 + \gamma_\mu)/2 \otimes U^\dagger_\mu(y) \delta_{x - \mu,y} \psi(y)
      PetscCall(DMPlexGetCone(dm, supp[2 * d + 0], &cone));
      PetscCall(PetscSectionGetOffset(s, cone[0], &yoff));
      PetscCall(PetscSectionGetOffset(sGauge, supp[2 * d + 0], &goff));
      PetscCall(DmobiusDhop(d, PETSC_FALSE, dag, &link[goff], &ua[yoff], &fa[xoff]));
      // Right edge -(1 - \gamma_\mu)/2 \otimes U_\mu(x) \delta_{x + \mu,y} \psi(y)
      PetscCall(DMPlexGetCone(dm, supp[2 * d + 1], &cone));
      PetscCall(PetscSectionGetOffset(s, cone[1], &yoff));
      PetscCall(PetscSectionGetOffset(sGauge, supp[2 * d + 1], &goff));
      PetscCall(DmobiusDhop(d, PETSC_TRUE, dag, &link[goff], &ua[yoff], &fa[xoff]));
    }
  }
  PetscCall(VecRestoreArray(f, &fa));
  PetscCall(VecRestoreArray(gauge, &link));
  PetscCall(VecRestoreArrayRead(u, &ua));
  PetscFunctionReturn(0);
}
static PetscErrorCode DmobiusInternal(DM dm, Vec u, Vec f,PetscBool dag)
{
  PetscFunctionBeginUser;
  Vec Din;
  Vec D5;
  PetscCall(DMCreateGlobalVector(dm, &D5));
  PetscCall(DMCreateGlobalVector(dm, &Din));

  MobiusDomainWallParameters *p = GetMobiusDomainWallParameters();

  PetscReal nrm;
  VecNorm(u,NORM_2,&nrm); printf(" u norm is %le\n",nrm);
  DmobiusDslash5(dm, u, D5, dag);
  VecNorm(D5,NORM_2,&nrm); printf(" D5 norm is %le\n",nrm);
  VecCopy(D5,Din);  
  VecNorm(Din,NORM_2,&nrm); printf(" Din norm is %le\n",nrm);

  // Din = p->b * u + p->c * D5;
  PetscScalar b = p->b;
  PetscScalar c = -p->c;
  VecAXPBY(Din,b,c,u); // y,a,b,x argument ordering (!)
  VecNorm(Din,NORM_2,&nrm); printf(" Din norm is %le\n",nrm);

  DmobiusDslash4(dm, Din, f ,dag);
  VecNorm(f,NORM_2,&nrm); printf(" Dout norm is %le\n",nrm);

  // f = f + D5
  PetscScalar scal = 1.0;
  VecAXPY(f,scal,D5);
  VecAXPY(f,scal,u);
  VecNorm(f,NORM_2,&nrm); printf(" Dout norm is %le\n",nrm);

  PetscCall(VecDestroy(&Din));
  PetscCall(VecDestroy(&D5));
  PetscFunctionReturn(0);
}
// Add MobiusDomainWall parameters
static PetscErrorCode Dmobius(DM dm, Vec u, Vec f)
{
  return DmobiusInternal(dm,u,f,PETSC_FALSE);
}
// Add MobiusDomainWall parameters
static PetscErrorCode DmobiusDag(DM dm, Vec u, Vec f)
{
  return DmobiusInternal(dm,u,f,PETSC_TRUE);
}
// Add MobiusDomainWall parameters
static PetscErrorCode DmobiusDagDmobius(DM dm, Vec u, Vec f)
{
  PetscFunctionBeginUser;
  Vec tmp;
  PetscCall(DMCreateGlobalVector(dm, &tmp));
  DmobiusInternal(dm,u,tmp,PETSC_FALSE);
  DmobiusInternal(dm,tmp,f,PETSC_TRUE);
  PetscCall(VecDestroy(&tmp));
  PetscFunctionReturn(0);
}

/*****************************************************************************************************************
 * Jacobians are empty
 *****************************************************************************************************************
 */
static PetscErrorCode CreateJacobian(DM dm, Mat *J)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode ComputeJacobian(DM dm, Vec u, Mat J)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}


#endif
