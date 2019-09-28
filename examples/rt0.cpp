//                                MFEM Runtime Example 0
//
// Compile with: make rt0

#include "mfem.hpp"
#include "general/dbg.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

double x0(const Vector &x)
{
   const double alpha = 8.0;
   const double r2 = alpha*(x(0)*x(0)+x(1)*x(1));
   return sin(r2)/r2;
}

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   int order = 2;
   int ref_levels = 2;
   const char *device_config = "cpu";
   const char *mesh_file = "../data/star.mesh";
   bool visualization = false;

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order", "Finite element order.");
   args.AddOption(&ref_levels, "-r", "--refine", "Refinement level.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string.");
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // Enable hardware devices such as GPUs, and programming models such as
   // CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   device.Print();

   // Read the mesh from the given mesh file. We can handle triangular,
   // quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   // the same code.
   Mesh mesh(mesh_file, 1, 1);
   const int dim = mesh.Dimension();
   for (int l = 0; l < ref_levels; l++) { mesh.UniformRefinement(); }


   L2_FECollection l2c(order, dim);
   FiniteElementSpace l2s(&mesh, &l2c);
   cout << "Number of finite element unknowns: " << l2s.GetTrueVSize() << endl;

   FunctionCoefficient x0_fct(x0);

   GridFunction x(&l2s), y(&l2s), z(&l2s);
   dbg("ProjectCoefficient");
   x.ProjectCoefficient(x0_fct);
   dbg("y = 1.0");
   y = 1.0;
   dbg("z = x.Add(1.0, y)");
   z = x.Add(1.0, y);

   // Send the solution by socket to a GLVis server.
   if (visualization)
   {
      int Wx = 0, Wy = 0; // window position
      int Ww = 640, Wh = 640; // window size
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << mesh << z
               << "window_geometry " << Wx << " " << Wy << " "
               << Ww << " " << Wh << " keys !mmAglii" << flush;
   }
   dbg("\n");
   return 0;
}
