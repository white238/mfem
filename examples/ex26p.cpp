#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Smooth spherical step function with radius t
double ball(double x, double y, double t) {
  const double alpha = 0.005;
  double r = sqrt(x * x + y * y);
  return -atan(2 * (r - t) / alpha);
}

double ball_func0(const Vector &coords) {
  const double x = coords(0);
  const double y = coords(1);
  const double x0 = 0.5;
  const double y0 = 0.5;

  return ball(x - x0, y - y0, 0.1);
}

class DiffusionMultigrid : public Multigrid {
private:
  ConstantCoefficient one;
  HypreBoomerAMG *amg;

public:
  DiffusionMultigrid(ParFiniteElementSpaceHierarchy &fespaces,
                     Array<int> &ess_bdr)
      : Multigrid(fespaces), one(1.0) {
    ConstructCoarseOperatorAndSolver(fespaces.GetFESpaceAtLevel(0), ess_bdr);

    for (int level = 1; level < fespaces.GetNumLevels(); ++level) {
      ConstructOperatorAndSmoother(fespaces.GetFESpaceAtLevel(level), ess_bdr);
    }
  }

  virtual ~DiffusionMultigrid() { delete amg; }

private:
  void ConstructBilinearForm(ParFiniteElementSpace &fespace,
                             Array<int> &ess_bdr, bool partial_assembly) {
    ParBilinearForm *form = new ParBilinearForm(&fespace);
    if (partial_assembly) {
      form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
    }
    form->AddDomainIntegrator(new DiffusionIntegrator(one));
    form->Assemble();
    bfs.Append(form);

    essentialTrueDofs.Append(new Array<int>());
    fespace.GetEssentialTrueDofs(ess_bdr, *essentialTrueDofs.Last());
  }

  void ConstructCoarseOperatorAndSolver(ParFiniteElementSpace &coarse_fespace,
                                        Array<int> &ess_bdr) {
    ConstructBilinearForm(coarse_fespace, ess_bdr, false);

    HypreParMatrix *hypreCoarseMat = new HypreParMatrix();
    bfs.Last()->FormSystemMatrix(*essentialTrueDofs.Last(), *hypreCoarseMat);

    amg = new HypreBoomerAMG(*hypreCoarseMat);
    amg->SetPrintLevel(-1);

    CGSolver *pcg = new CGSolver(MPI_COMM_WORLD);
    pcg->SetPrintLevel(-1);
    pcg->SetMaxIter(10);
    pcg->SetRelTol(sqrt(1e-4));
    pcg->SetAbsTol(0.0);
    pcg->SetOperator(*hypreCoarseMat);
    pcg->SetPreconditioner(*amg);

    AddLevel(hypreCoarseMat, pcg, true, true);
  }

  void ConstructOperatorAndSmoother(ParFiniteElementSpace &fespace,
                                    Array<int> &ess_bdr) {
    ConstructBilinearForm(fespace, ess_bdr, true);

    OperatorPtr opr;
    opr.SetType(Operator::ANY_TYPE);
    bfs.Last()->FormSystemMatrix(*essentialTrueDofs.Last(), opr);
    opr.SetOperatorOwner(false);

    Vector diag(fespace.GetTrueVSize());
    bfs.Last()->AssembleDiagonal(diag);

    Solver *smoother = new OperatorChebyshevSmoother(
        opr.Ptr(), diag, *essentialTrueDofs.Last(), 2,
        fespace.GetParMesh()->GetComm());

    AddLevel(opr.Ptr(), smoother, true, true);
  }
};

int main(int argc, char *argv[]) {

  int num_procs, myid;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  const char *mesh_file = "../data/inline-quad.mesh";
  int geometric_refinements = 0;
  const char *device_config = "cpu";
  bool visualization = true;
  const int order = 2;

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(
      &geometric_refinements, "-gr", "--geometric-refinements",
      "Number of geometric refinements done prior to order refinements.");
  args.AddOption(&device_config, "-d", "--device",
                 "Device configuration string, see Device::Configure().");
  args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                 "--no-visualization",
                 "Enable or disable GLVis visualization.");
  args.Parse();
  if (!args.Good()) {
    if (myid == 0) {
      args.PrintUsage(cout);
    }
    MPI_Finalize();
    return 1;
  }
  if (myid == 0) {
    args.PrintOptions(cout);
  }

  Device device(device_config);
  if (myid == 0) {
    device.Print();
  }

  Mesh *mesh = new Mesh(mesh_file, 1, 1);
  mesh->EnsureNCMesh(true);
  int dim = mesh->Dimension();

  ParMesh *coarse_pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
  delete mesh;
  {
    int par_ref_levels = 2;
    for (int l = 0; l < par_ref_levels; l++) {
      coarse_pmesh->UniformRefinement();
    }
  }

  // Fallback coarse space (fixed, never changes)
  FiniteElementCollection *fec = new H1_FECollection(order, dim);
  ParFiniteElementSpace *coarse_fespace =
      new ParFiniteElementSpace(coarse_pmesh, fec);

  ParGridFunction *x_distribution = nullptr;

  // Fake outer loop (optimization or similar)
  {
    // Make a deep copy of the mesh for refinement
    ParMesh *pmesh = new ParMesh(*coarse_pmesh);

    // Create an fespace on the coarse mesh to update during AMR
    ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);

    ParFiniteElementSpaceHierarchy *fespaces =
        new ParFiniteElementSpaceHierarchy(pmesh, fespace, true, true);

    printf("Coarse fespace size: %d\n", fespace->GetTrueVSize());

    // Do AMR based on fluxes of a discrete distribution, starting from the
    // coarsest mesh
    x_distribution = new ParGridFunction(fespace);
    FunctionCoefficient distribution0_coeff(ball_func0);
    x_distribution->ProjectCoefficient(distribution0_coeff);

    BilinearFormIntegrator *integ = new DiffusionIntegrator;

    L2_FECollection flux_fec(order, dim);
    ParFiniteElementSpace *flux_fes = new ParFiniteElementSpace(pmesh, &flux_fec, 2);
    RT_FECollection smooth_flux_fec(order - 1, dim);
    ParFiniteElementSpace *smooth_flux_fes = new ParFiniteElementSpace(pmesh, &smooth_flux_fec);
    L2ZienkiewiczZhuEstimator estimator(*integ, *x_distribution, flux_fes,
                                        smooth_flux_fes);

    const double max_elem_error = 1.0e-3;
    ThresholdRefiner refiner(estimator);
    refiner.SetTotalErrorFraction(0.0); // use purely local threshold
    refiner.SetLocalErrorGoal(max_elem_error);
    refiner.PreferConformingRefinement();
    refiner.SetNCLimit(1);

    const int max_amr_cycles = 2;
    int amr_cycle = 0;

    while (!refiner.Stop() && amr_cycle < max_amr_cycles) {
      amr_cycle++;
      printf("AMR iteration: %d\n", amr_cycle);

      ParMesh *refined_pmesh = new ParMesh(*pmesh);
      ParFiniteElementSpace *refined_fespace = new ParFiniteElementSpace(refined_pmesh, fec);

      refiner.Reset();
      refiner.Apply(*refined_pmesh);

      // if (refined_pmesh->Nonconforming()) {
      //   refined_pmesh->Rebalance();
      // }

      refined_fespace->Update();
      Operator *P = new TransferOperator(*fespace, *refined_fespace);
      fespaces->AddLevel(refined_pmesh, refined_fespace, P, true, true, true);
      refined_fespace->UpdatesFinished();

      printf("Level %d fespace size: %d\n", amr_cycle, fespaces->GetFESpaceAtLevel(amr_cycle).GetTrueVSize());

      pmesh = refined_pmesh;
      fespace = refined_fespace;

      flux_fes = new ParFiniteElementSpace(pmesh, &flux_fec, 2);
      smooth_flux_fes = new ParFiniteElementSpace(pmesh, &smooth_flux_fec);

      estimator.SetFluxSpaces(flux_fes, smooth_flux_fes);

      x_distribution->SetSpace(fespace);
      x_distribution->ProjectCoefficient(distribution0_coeff);
    }

    HYPRE_Int size = fespaces->GetFinestFESpace().GlobalTrueVSize();
    if (myid == 0) {
      cout << "Number of finite element unknowns: " << size << endl;
    }

    ParLinearForm *b = new ParLinearForm(&fespaces->GetFinestFESpace());
    ConstantCoefficient one(1.0);
    b->AddDomainIntegrator(new DomainLFIntegrator(one));
    b->Assemble();

    ParGridFunction x(&fespaces->GetFinestFESpace());
    x = 0.0;

    Array<int> ess_bdr(pmesh->bdr_attributes.Max());
    if (pmesh->bdr_attributes.Size()) {
      ess_bdr = 1;
    }

    DiffusionMultigrid *M = new DiffusionMultigrid(*fespaces, ess_bdr);
    M->SetCycleType(Multigrid::CycleType::VCYCLE, 1, 1);

    OperatorPtr A;
    Vector X, B;
    M->FormFineLinearSystem(x, *b, A, X, B);

    CGSolver cg(MPI_COMM_WORLD);
    // cg.iterative_mode = false;
    cg.SetAbsTol(0.0);
    cg.SetRelTol(1e-8);
    cg.SetMaxIter(2000);
    cg.SetPrintLevel(1);
    cg.SetOperator(*A);
    cg.SetPreconditioner(*M);
    cg.Mult(B, X);

    M->RecoverFineFEMSolution(X, *b, x);

    if (visualization) {
      char vishost[] = "localhost";
      int visport = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n"
               << *fespaces->GetFinestFESpace().GetParMesh() << x << flush;
      // sol_sock << "solution\n" << *pmesh << x_distribution << flush;
    }

  }


  MPI_Finalize();
  return 0;
}
