// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include "rts.hpp"
#include "forall.hpp"

#include <functional>
#include <list>
#include <string>
#include <unordered_set>

namespace mfem
{

// Initialize the unique global Runtime variable.
Runtime Runtime::runtime_singleton;

Runtime::~Runtime()
{
   if (i_am_this) { /* only *this will do something here */}
}

void Runtime::Setup()
{
   InOutClear();
   kernels.clear();
   // Copy all data members from the global 'runtime_singleton' into '*this'.
   std::memcpy(this, &Get(), sizeof(Runtime));
   i_am_this = true;
}

void Runtime::Print(std::ostream &out) { dbg(""); }

void Runtime::RW_(void *p, const bool use_dev, int m)
{
   if (!use_dev) { return; }
   if (!Device::IsEnabled()) { return; }
   switch (m)
   {
      case 0:
      {
         Get().in.push_back(p);
         //printf("\n\033[32m%p:%d \033[m", p, m);
         break;
      }
      case 1:
      {
         Get().out.push_back(p);
         //printf("\n\033[35m%p:%d \033[m", p, m);
         //mfem_backtrace();
         break;
      }
      case 2:
      {
         Get().inout.push_back(p);
         //printf("\n\033[31m%p:%d \033[m", p, m);
         break;
      }
      default: mfem_error();
   }
   //fflush(0);
}

void Runtime::RW_(const void *p, const bool use_dev, int m)
{
   return RW_(const_cast<void*>(p), use_dev, m);
}


void Runtime::Sync() { printf("\033[33mS\033[m"); fflush(0); }

void Runtime::Kernel(const bool use_dev,
                     const char *file, const int line,
                     const char *function, const char *s_body)
{
   if (!use_dev) { InOutClear(); return; }
   if (!Device::IsEnabled()) { InOutClear(); return; }
   //printf("\n\033[33m[kernel] '%s' (%s:%d)\033[m", function, file, line);
   //printf("\n\033[33m[kernel] %s\033[m", s_body);

   void_ptr_l &in = Get().in;
   void_ptr_l &out = Get().out;
   void_ptr_l &inout = Get().inout;
   kernel_l &kernels = Get().kernels;

   // Add kernel if we haven't seen him yet
   kernel_l::const_iterator k;
   std::stringstream kernal_name_ss;
   kernal_name_ss << "[" << file;
   kernal_name_ss << ":" << line;
   kernal_name_ss << ":" << function << "]";
   std::string kernel_s(kernal_name_ss.str());
   const char *kernel_name = kernel_s.c_str();
   auto it = std::find(kernels.begin(), kernels.end(), kernel_name);
   if (it == kernels.end())
   {
      printf("\n\n\033[33m[kernel] New kernel (%s)\033[m", kernel_name);
      kernels.push_back(kernel_s);
   }
   else
   {
      printf("\n\n\033[33m[kernel] Known kernel (%s)\033[m", kernel_name);
      /*for (k=kernels.begin(); k != kernels.end(); ++k)
      { printf("\n\t\033[33m[kernel] Known kernels are: \033[32m%s\033[m", k->c_str()); }
      */
   }

   void_ptr_l::const_iterator i;
   for (i=in.begin(); i != in.end(); ++i)
   { printf("\n\t\033[33m[kernel] \033[32m%p\033[m", *i); }

   for (i=out.begin(); i != out.end(); ++i)
   { printf("\n\t\033[33m[kernel] \033[35m%p\033[m", *i); }

   for (i=inout.begin(); i != inout.end(); ++i)
   { printf("\n\t\033[33m[kernel] \033[31m%p\033[m", *i); }

   InOutClear();
}

void Runtime::Memcpy(const char *function, void *dst, const void *src,
                     size_t bytes)
{
   printf("\n\033[33m[Memcpy] %s \033[32m%p\033[m => \033[35m%p\033[m (%ld bytes)\033[m",
          function, src, dst, bytes);
   InOutClear();
}

} // mfem
