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
#ifndef MFEM_RTS_HPP
#define MFEM_RTS_HPP

#include <cassert>
#include <list>
//#include <algorithm>
#include <unordered_map>

#include "dbg.hpp"
#include "error.hpp"
#include "globals.hpp"

namespace mfem
{
class Runtime
{

private:
	// Kernel struct
	struct kernel_t
	{
		const int rank;
		const char *file;
		const int line;
		const char *function;
		const char *body;
		const char *hash;
		bool operator<(const kernel_t &k)
		{
			MFEM_VERIFY(k.hash && hash,"")
			return strcmp(hash,k.hash);
		}
		bool operator==(const char *h)
		{
			MFEM_VERIFY(hash && h,"")
			return strcmp(h,hash)==0;
		}
		bool operator==(const kernel_t &k) const
		{
			MFEM_VERIFY(k.hash && hash,"")
			return strcmp(hash,k.hash)==0;
		}
	};

	// Address struct
	struct address_t
	{
		const int rank;
		const void *address;
		const kernel_t &kernel;
		bool operator<(address_t &that)
		{
			if (this->rank != that.rank) { return this->rank < that.rank; }
			return this->address < that.address;
		}
		bool operator<(const address_t &that) const
		{
			if (this->rank != that.rank) { return this->rank < that.rank; }
			return this->address < that.address;
		}
		bool operator==(address_t &that)
		{
			return (rank == that.rank && address == that.address);
		}
		bool operator==(const address_t &that) const
		{
			return (rank == that.rank && address == that.address);
		}
	};

	typedef std::list<kernel_t> kernel_l;
	typedef kernel_l::const_iterator k_it;

	typedef std::list<address_t> inputs_l;
	typedef inputs_l::const_iterator a_it;
	inputs_l input_address, output_address;

	typedef std::list<void*> ptr_l;
	typedef ptr_l::const_iterator p_it;
	ptr_l in, out;
	std::unordered_map<const void*,const char*> names;

	int rank;
	kernel_l kernels;

	static Runtime runtime_singleton;

	bool ready = false;
	bool record = false;
	bool i_am_this = false;

private:
	void Setup_();
	void DumpGraph_();
	Runtime(Runtime const&);
	void operator=(Runtime const&);
	static Runtime& RTS() { return runtime_singleton; }

private:
	int GetRank_();
	void IncRank_();
	void InOutClear_();
	void Start_();
	void For_();
	void Return_();
	void Break_();
	void Loop_();
	void Cond_(const char *test);
	void Stop_();
	void RW_(void *p, const bool use_dev, int m=0);
	void RW_(const void *p, const bool use_dev, int m=0);
	void InOutPushBack_();
	template <typename T> T Name_(const char *name, T adrs)
	{
		if (ready) { return adrs;}
		if (!record) { return adrs;}
		dbg("\033[7;1;37mRuntime::Name Insert %p %s %d", adrs, name, rank);
		auto name_it = names.find(adrs);
		if (name_it == names.end())
		{
			auto res = names.emplace(adrs, strdup(name));
			if (res.second == false) // was already in the map
			{
				dbg("\033[7;1;31mImpossible!");
			}
		}
		else
		{
			dbg("\033[7;1;31mRuntime::Name adrs %p was %s!", adrs, name_it->second);
		}
		return adrs;
	}
	const char *Name_(const void *adrs/*, const int rank*/);

public:
	Runtime():ready(false), i_am_this(false) { RTS().Setup_(); }
	~Runtime();

public:
	static inline bool IsReady() { return RTS().ready; }
	static inline void Start() { RTS().Start_(); }
	static inline void For() { RTS().For_(); }
	static inline void Loop() { RTS().Loop_(); }
	static inline void Stop() { RTS().Stop_(); }
	static inline void Break() { RTS().Break_(); }
	static inline void Return() { RTS().Return_(); }
	static inline void Cond(const char *test) { RTS().Cond_(test); }
	static inline void InOutClear() { RTS().InOutClear_(); }
	static inline void InOutPushBack() { RTS().InOutPushBack_(); }
	template <typename T>
	static inline T Name(const char *name, T adrs) { return RTS().Name_(name, adrs); }
	static void Kernel(const bool, const char*, const int, const char*,
							 const char*, const int N=0,
							 const int X=0, const int Y=0, const int Z=0);
	static void Memcpy(const char*, void*, const void*, size_t);

public:
	template <typename T> static const T *R(const T *p, const bool use_dev)
	{ RTS().RW_(p,use_dev,0); return p; }
	template <typename T> static T *W(T *p, const bool use_dev)
	{ RTS().RW_(p,use_dev,1); return p; }
	template <typename T> static T *RW(T *p, const bool use_dev)
	{ RTS().RW_(p,use_dev,2); return p; }
};

} // mfem

#endif // MFEM_RTS_HPP
