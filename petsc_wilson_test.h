#ifndef _PETSC_WILSON_TEST_H_
#define _PETSC_WILSON_TEST_H_
/*
  Test the action of the Wilson operator in the free field case U = I,

    \eta(x) = D_W(x - y) \psi(y)

  The Wilson operator is a convolution for the free field, so we can check that by the convolution theorem

    \hat\eta(x) = \mathcal{F}(D_W(x - y) \psi(y))
                = \hat D_W(p) \mathcal{F}\psi(p)

  The Fourier series for the Wilson operator is

    M + \sum_\mu 2 \sin^2(p_\mu / 2) + i \gamma_\mu \sin(p_\mu)
*/
static PetscErrorCode TestFreeField(DM dm)
{
  PetscSection       s;
  Mat                FT;
  Vec                psi, psiHat;
  Vec                eta, etaHat;
  Vec                DHat; // The product \hat D_w \hat psi
  PetscRandom        r;
  const PetscScalar *psih;
  PetscScalar       *dh;
  PetscReal         *coef, nrm;
  const PetscInt    *extent, Nc = 12;
  PetscInt           dim, V     = 1, vStart, vEnd;
  PetscContainer     c;
  PetscBool          constRhs = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-const_rhs", &constRhs, NULL));

  PetscCall(SetGauge_Identity(dm));
  PetscCall(DMGetLocalSection(dm, &s));
  PetscCall(DMGetGlobalVector(dm, &psi));
  PetscCall(PetscObjectSetName((PetscObject)psi, "psi"));
  PetscCall(DMGetGlobalVector(dm, &psiHat));
  PetscCall(PetscObjectSetName((PetscObject)psiHat, "psihat"));
  PetscCall(DMGetGlobalVector(dm, &eta));
  PetscCall(PetscObjectSetName((PetscObject)eta, "eta"));
  PetscCall(DMGetGlobalVector(dm, &etaHat));
  PetscCall(PetscObjectSetName((PetscObject)etaHat, "etahat"));
  PetscCall(DMGetGlobalVector(dm, &DHat));
  PetscCall(PetscObjectSetName((PetscObject)DHat, "Dhat"));
  PetscCall(PetscRandomCreate(PETSC_COMM_SELF, &r));
  PetscCall(PetscRandomSetType(r, PETSCRAND48));
  if (constRhs) PetscCall(VecSet(psi, 1.));
  else PetscCall(VecSetRandom(psi, r));
  PetscCall(PetscRandomDestroy(&r));

  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  PetscCall(PetscObjectQuery((PetscObject)dm, "_extent", (PetscObject *)&c));
  PetscCall(PetscContainerGetPointer(c, (void **)&extent));
  PetscCall(MatCreateFFT(PetscObjectComm((PetscObject)dm), dim, extent, MATFFTW, &FT));

  PetscCall(PetscMalloc1(dim, &coef));
  for (PetscInt d = 0; d < dim; ++d) {
    coef[d] = 2. * PETSC_PI / extent[d];
    V *= extent[d];
  }
  PetscCall(Dwilson(dm, psi, eta));
  PetscCall(VecViewFromOptions(eta, NULL, "-psi_view"));
  PetscCall(VecViewFromOptions(eta, NULL, "-eta_view"));
  PetscCall(ComputeFFT(FT, Nc, psi, psiHat));
  PetscCall(VecScale(psiHat, 1. / V));
  PetscCall(ComputeFFT(FT, Nc, eta, etaHat));
  PetscCall(VecScale(etaHat, 1. / V));
  PetscCall(VecGetArrayRead(psiHat, &psih));
  PetscCall(VecGetArray(DHat, &dh));
  for (PetscInt v = vStart; v < vEnd; ++v) {
    PetscScalar tmp[12], tmp1 = 0.;
    PetscInt    dof, off;

    PetscCall(PetscSectionGetDof(s, v, &dof));
    PetscCall(PetscSectionGetOffset(s, v, &off));
    for (PetscInt d = 0, prod = 1; d < dim; prod *= extent[d], ++d) {
      const PetscInt idx = (v / prod) % extent[d];

      tmp1 += 2. * PetscSqr(PetscSinReal(0.5 * coef[d] * idx));
      for (PetscInt i = 0; i < dof; ++i) tmp[i] = PETSC_i * PetscSinReal(coef[d] * idx) * psih[off + i];
      for (PetscInt c = 0; c < 3; ++c) GammaMuTimesSpinor(d, 3, &tmp[c]);
      for (PetscInt i = 0; i < dof; ++i) dh[off + i] += tmp[i];
    }
    for (PetscInt i = 0; i < dof; ++i) dh[off + i] += (M + tmp1) * psih[off + i];
  }
  PetscCall(VecRestoreArrayRead(psiHat, &psih));
  PetscCall(VecRestoreArray(DHat, &dh));

  {
    Vec     *etaComp, *DComp;
    PetscInt n, N;

    PetscCall(PetscMalloc2(Nc, &etaComp, Nc, &DComp));
    PetscCall(VecGetLocalSize(etaHat, &n));
    PetscCall(VecGetSize(etaHat, &N));
    for (PetscInt i = 0; i < Nc; ++i) {
      const char *vtype;

      // HACK: Make these from another DM up front
      PetscCall(VecCreate(PetscObjectComm((PetscObject)etaHat), &etaComp[i]));
      PetscCall(VecGetType(etaHat, &vtype));
      PetscCall(VecSetType(etaComp[i], vtype));
      PetscCall(VecSetSizes(etaComp[i], n / Nc, N / Nc));
      PetscCall(VecDuplicate(etaComp[i], &DComp[i]));
    }
    PetscCall(VecStrideGatherAll(etaHat, etaComp, INSERT_VALUES));
    PetscCall(VecStrideGatherAll(DHat, DComp, INSERT_VALUES));
    for (PetscInt i = 0; i < Nc; ++i) {
      if (!i) {
        PetscCall(VecViewFromOptions(etaComp[i], NULL, "-etahat_view"));
        PetscCall(VecViewFromOptions(DComp[i], NULL, "-dhat_view"));
      }
      PetscCall(VecAXPY(etaComp[i], -1., DComp[i]));
      PetscCall(VecNorm(etaComp[i], NORM_INFINITY, &nrm));
      PetscCall(PetscPrintf(PETSC_COMM_SELF, "  Slice %" PetscInt_FMT ": %g\n", i, (double)nrm));
    }
    PetscCall(VecStrideScatterAll(etaComp, etaHat, INSERT_VALUES));
    for (PetscInt i = 0; i < Nc; ++i) {
      PetscCall(VecDestroy(&etaComp[i]));
      PetscCall(VecDestroy(&DComp[i]));
    }
    PetscCall(PetscFree2(etaComp, DComp));
    PetscCall(VecNorm(etaHat, NORM_INFINITY, &nrm));
    PetscCheck(nrm < PETSC_SMALL, PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Free field test failed: %g", (double)nrm);
  }

  PetscCall(PetscFree(coef));
  PetscCall(MatDestroy(&FT));
  PetscCall(DMRestoreGlobalVector(dm, &psi));
  PetscCall(DMRestoreGlobalVector(dm, &psiHat));
  PetscCall(DMRestoreGlobalVector(dm, &eta));
  PetscCall(DMRestoreGlobalVector(dm, &etaHat));
  PetscCall(DMRestoreGlobalVector(dm, &DHat));
  PetscFunctionReturn(0);
}
#endif
