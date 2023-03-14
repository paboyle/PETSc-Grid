#include <Grid/Grid.h>

#ifndef _PETSC_GRID_H
#define _PETSC_GRID_H

NAMESPACE_BEGIN(Grid);
template<class vobj>
int PetscToGrid(DM dm,Vec psi,Lattice<vobj> &g_psi)
{
  typedef typename vobj::scalar_object sobj;

  GridBase *grid = g_psi.Grid();
  uint64_t lsites = grid->lSites();
  std::vector<sobj> scalardata(lsites);
  
  const PetscScalar *psih;
  PetscInt           dim, vStart, vEnd;
  PetscSection       s;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetLocalSection(dm, &s));
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  PetscCall(VecGetArrayRead(psi, &psih));

  Integer idx=0  ;
  for (PetscInt v = vStart ; v < vEnd; ++v,++idx) {
    PetscInt    dof, off;
    PetscCall(PetscSectionGetDof(s, v, &dof));
    PetscCall(PetscSectionGetOffset(s, v, &off));
    ComplexD *g_p = (ComplexD *)& scalardata[idx];
    for(int d=0;d<12;d++){
      double re =PetscRealPart(psih[off + d]);
      double im =PetscImaginaryPart(psih[off + d]);
      g_p[d]=ComplexD(re,im);
    }
  }
  assert(idx==lsites);
  vectorizeFromLexOrdArray(scalardata,g_psi);    
  PetscFunctionReturn(0);
}
template<class vobj>
int GridToPetsc(DM dm,Vec psi,Lattice<vobj> &g_psi)
{
  typedef typename vobj::scalar_object sobj;

  GridBase *grid = g_psi.Grid();
  uint64_t lsites = grid->lSites();
  std::vector<sobj> scalardata(lsites);
  unvectorizeToLexOrdArray(scalardata,g_psi);    

  PetscScalar *psih;
  PetscInt           dim, vStart, vEnd;
  PetscSection       s;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetLocalSection(dm, &s));
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  PetscCall(VecGetArray(psi, &psih));
  Integer idx=0  ;
  for (PetscInt v = vStart ; v < vEnd; ++v,++idx) {
    PetscInt    dof, off;
    PetscCall(PetscSectionGetDof(s, v, &dof));
    PetscCall(PetscSectionGetOffset(s, v, &off));
    ComplexD *g_p = (ComplexD *)& scalardata[idx];
    for(int d=0;d<12;d++){
      double re =real(g_p[d]);
      double im =imag(g_p[d]);
      psih[off + d] = re + im * PETSC_i;
    }
  }
  assert(idx==lsites);
  PetscFunctionReturn(0);
}
static PetscErrorCode SetGauge_Grid(DM dm, LatticeGaugeField & Umu)
{
  typedef typename LatticeGaugeField::scalar_object sobj;
  GridBase *grid = Umu.Grid();
  uint64_t lsites = grid->lSites();
  std::vector<sobj> scalardata(lsites);
  unvectorizeToLexOrdArray(scalardata,Umu);
  
  DM           auxDM;
  Vec          auxVec;
  PetscSection s;
  PetscScalar  *id;
  PetscInt     eStart, eEnd;

  PetscFunctionBegin;
  PetscCall(DMGetAuxiliaryVec(dm, NULL, 0, 0, &auxVec));
  PetscCall(VecGetDM(auxVec, &auxDM));
  PetscCall(DMGetLocalSection(auxDM, &s));
  PetscCall(DMPlexGetDepthStratum(dm, 1, &eStart, &eEnd));
  int j=0;
  ColourMatrixD *Grid_p = (ColourMatrixD *)&scalardata[0];
  // Ordering of edges ???
  for (PetscInt i = eStart; i < eEnd; ++i) {
    ColourMatrixD U = Grid_p[j];
    id = (PetscScalar *) &U;
    PetscCall(VecSetValuesSection(auxVec, s, i, id, INSERT_VALUES));
    j++;
  }
  std::cout << " Set "<<j<<" gauge links "<<std::endl;
  std::cout << " Grid lsites "<<lsites<<std::endl;
  PetscCall(VecViewFromOptions(auxVec, NULL, "-gauge_view"));
  PetscFunctionReturn(0);
}
static PetscErrorCode SetGauge_Grid5D(DM dm, LatticeGaugeField & Umu, PetscInt Ls, PetscReal m)
{
  typedef typename LatticeGaugeField::scalar_object sobj;
  GridBase *grid = Umu.Grid();
  uint64_t lsites = grid->lSites();
  std::vector<sobj> scalardata(lsites);
  unvectorizeToLexOrdArray(scalardata,Umu);

  PetscScalar  idm[9] = {-m, 0., 0., 0., -m, 0., 0., 0., -m};
  PetscScalar  id[9]  = { 1., 0., 0., 0., 1., 0., 0., 0., 1.};
  
  DM           auxDM;
  Vec          auxVec;
  PetscSection s;
  PetscScalar  *up;
  PetscInt     eStart, eEnd;

  PetscFunctionBegin;
  PetscCall(DMGetAuxiliaryVec(dm, NULL, 0, 0, &auxVec));
  PetscCall(VecGetDM(auxVec, &auxDM));
  PetscCall(DMGetLocalSection(auxDM, &s));
  PetscCall(DMPlexGetDepthStratum(dm, 1, &eStart, &eEnd));
  ColourMatrixD *Grid_p = (ColourMatrixD *)&scalardata[0];

  for (PetscInt i = eStart; i < eEnd; ++i) {
    int d = i % 5;
    int s5=i/5;
    int ss=s5%Ls;
    int s4=s5/Ls;
    if ( d==0 ) { // Unit link in s-dim
      if ( ss == Ls-1 ) {
	PetscCall(VecSetValuesSection(auxVec, s, i, idm, INSERT_VALUES));
      } else { 
	PetscCall(VecSetValuesSection(auxVec, s, i, id, INSERT_VALUES));
      }
    } else {
      int jj=d-1 + 4*s4;
      ColourMatrixD U = Grid_p[jj];
      up = (PetscScalar *) &U;
      PetscCall(VecSetValuesSection(auxVec, s, i, up, INSERT_VALUES));
    }
  }
  PetscCall(VecViewFromOptions(auxVec, NULL, "-gauge_view"));
  PetscFunctionReturn(0);
}
NAMESPACE_END(Grid);
#endif
