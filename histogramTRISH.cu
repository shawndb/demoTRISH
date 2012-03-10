/*-----------------------------------------------------------------------------
   Name: histogramTRISH.cu
   Desc: Implements 256-way binning histogram algorithm on GPU
   
   Disclaimer:
      This software is distributed in the hope that it will be useful,
      but WITHOUT ANY WARRANTY; without even the implied warranty of
      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
-----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
  Includes
-----------------------------------------------------------------------------*/

// System Includes
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// CUDA Includes
#include <cutil_inline.h>

// Local Includes
#include "histogram_common.h"


/*-----------------------------------------------------------------------------
  Helper Templates
-----------------------------------------------------------------------------*/

/*---------------------------------------------------------
  Name:   SetArray_BlockSeq
  Desc:   Sets elements in array to specified value
  Note:   Uses "Block Sequential" access pattern
 ---------------------------------------------------------*/

template <  typename valT,		// Underlying value type
            uint BlockSize,    // ThreadPerBlock
            uint nSafePasses,  // Number of safe passes
            uint nLeftOver,    // Number of left over elements
            uint maxSize >     // Max Size of array
__device__ __forceinline__
void SetArray_BlockSeq
( 
   valT * basePtr,      // IN/OUT - array to set to 'set' value
   valT   toSet         // IN - value to set array elements 'to'
) 
{
   // Get 'per thread' pointer
   valT * setPtr = basePtr + threadIdx.x;

		// Initialize as many elements as we
		// safely can with no range checking
	if (nSafePasses >=  1u) { setPtr[( 0u * BlockSize)] = toSet; }
	if (nSafePasses >=  2u) { setPtr[( 1u * BlockSize)] = toSet; }
	if (nSafePasses >=  3u) { setPtr[( 2u * BlockSize)] = toSet; }
	if (nSafePasses >=  4u) { setPtr[( 3u * BlockSize)] = toSet; }
	if (nSafePasses >=  5u) { setPtr[( 4u * BlockSize)] = toSet; }
	if (nSafePasses >=  6u) { setPtr[( 5u * BlockSize)] = toSet; }
	if (nSafePasses >=  7u) { setPtr[( 6u * BlockSize)] = toSet; }
	if (nSafePasses >=  8u) { setPtr[( 7u * BlockSize)] = toSet; }
	if (nSafePasses >=  9u) { setPtr[( 8u * BlockSize)] = toSet; }
	if (nSafePasses >= 10u) { setPtr[( 9u * BlockSize)] = toSet; }
	if (nSafePasses >= 11u) { setPtr[(10u * BlockSize)] = toSet; }
	if (nSafePasses >= 12u) { setPtr[(11u * BlockSize)] = toSet; }
	if (nSafePasses >= 13u) { setPtr[(12u * BlockSize)] = toSet; }
	if (nSafePasses >= 14u) { setPtr[(13u * BlockSize)] = toSet; }
	if (nSafePasses >= 15u) { setPtr[(14u * BlockSize)] = toSet; }
	if (nSafePasses >= 16u) { setPtr[(15u * BlockSize)] = toSet; }
	if (nSafePasses >= 17u) { setPtr[(16u * BlockSize)] = toSet; }
	if (nSafePasses >= 18u) { setPtr[(17u * BlockSize)] = toSet; }
	if (nSafePasses >= 19u) { setPtr[(18u * BlockSize)] = toSet; }
	if (nSafePasses >= 20u) { setPtr[(19u * BlockSize)] = toSet; }
	if (nSafePasses >= 21u) { setPtr[(20u * BlockSize)] = toSet; }
	if (nSafePasses >= 22u) { setPtr[(21u * BlockSize)] = toSet; }
	if (nSafePasses >= 23u) { setPtr[(22u * BlockSize)] = toSet; }
	if (nSafePasses >= 24u) { setPtr[(23u * BlockSize)] = toSet; }
	if (nSafePasses >= 25u) { setPtr[(24u * BlockSize)] = toSet; }
	if (nSafePasses >= 26u) { setPtr[(25u * BlockSize)] = toSet; }
	if (nSafePasses >= 27u) { setPtr[(26u * BlockSize)] = toSet; }
	if (nSafePasses >= 28u) { setPtr[(27u * BlockSize)] = toSet; }
	if (nSafePasses >= 29u) { setPtr[(28u * BlockSize)] = toSet; }
	if (nSafePasses >= 30u) { setPtr[(29u * BlockSize)] = toSet; }
	if (nSafePasses >= 31u) { setPtr[(30u * BlockSize)] = toSet; }
	if (nSafePasses >= 32u) { setPtr[(31u * BlockSize)] = toSet; }
	if (nSafePasses >= 33u) { setPtr[(32u * BlockSize)] = toSet; }
	if (nSafePasses >= 34u) { setPtr[(33u * BlockSize)] = toSet; }
	if (nSafePasses >= 35u) { setPtr[(34u * BlockSize)] = toSet; }
	if (nSafePasses >= 36u) { setPtr[(35u * BlockSize)] = toSet; }
	if (nSafePasses >= 37u) { setPtr[(36u * BlockSize)] = toSet; }
	if (nSafePasses >= 38u) { setPtr[(37u * BlockSize)] = toSet; }
	if (nSafePasses >= 39u) { setPtr[(38u * BlockSize)] = toSet; }
	if (nSafePasses >= 40u) { setPtr[(39u * BlockSize)] = toSet; }
	if (nSafePasses >= 41u) { setPtr[(40u * BlockSize)] = toSet; }
	if (nSafePasses >= 42u) { setPtr[(41u * BlockSize)] = toSet; }
	if (nSafePasses >= 43u) { setPtr[(42u * BlockSize)] = toSet; }
	if (nSafePasses >= 44u) { setPtr[(43u * BlockSize)] = toSet; }
	if (nSafePasses >= 45u) { setPtr[(44u * BlockSize)] = toSet; }
	if (nSafePasses >= 46u) { setPtr[(45u * BlockSize)] = toSet; }
	if (nSafePasses >= 47u) { setPtr[(46u * BlockSize)] = toSet; }
	if (nSafePasses >= 48u) { setPtr[(47u * BlockSize)] = toSet; }
	if (nSafePasses >= 49u) { setPtr[(48u * BlockSize)] = toSet; }
	if (nSafePasses >= 50u) { setPtr[(49u * BlockSize)] = toSet; }
	if (nSafePasses >= 51u) { setPtr[(50u * BlockSize)] = toSet; }
	if (nSafePasses >= 52u) { setPtr[(51u * BlockSize)] = toSet; }
	if (nSafePasses >= 53u) { setPtr[(52u * BlockSize)] = toSet; }
	if (nSafePasses >= 54u) { setPtr[(53u * BlockSize)] = toSet; }
	if (nSafePasses >= 55u) { setPtr[(54u * BlockSize)] = toSet; }
	if (nSafePasses >= 56u) { setPtr[(55u * BlockSize)] = toSet; }
	if (nSafePasses >= 57u) { setPtr[(56u * BlockSize)] = toSet; }
	if (nSafePasses >= 58u) { setPtr[(57u * BlockSize)] = toSet; }
	if (nSafePasses >= 59u) { setPtr[(58u * BlockSize)] = toSet; }
	if (nSafePasses >= 60u) { setPtr[(59u * BlockSize)] = toSet; }
	if (nSafePasses >= 61u) { setPtr[(60u * BlockSize)] = toSet; }
	if (nSafePasses >= 62u) { setPtr[(61u * BlockSize)] = toSet; }
	if (nSafePasses >= 63u) { setPtr[(62u * BlockSize)] = toSet; }
	if (nSafePasses >= 64u) { setPtr[(63u * BlockSize)] = toSet; }
	if (nSafePasses >= 65u) { setPtr[(64u * BlockSize)] = toSet; }
	if (nSafePasses >= 66u) { setPtr[(65u * BlockSize)] = toSet; }

	// Set any 'left over' values with range checking
	if (nLeftOver > 0u)
	{ 
		uint idx = (nSafePasses * BlockSize) + threadIdx.x;
		if (idx < maxSize)
		{
			basePtr[idx] = toSet;
		}
	}
}


/*---------------------------------------------------------
  Name:   SetArray_WarpSeq
  Desc:   Sets elements in array to specified value
  Note:   Uses "Warp Sequential" access pattern
 ---------------------------------------------------------*/

template <  typename valT,		// Underlying value type
            uint WarpSize,     // Threads per Warp
            uint nSafePasses,  // Number of safe passes (warps per subsection)
            uint nLeftOver,    // Number of left over elements
            uint maxSize >     // Max Size of array
__device__ __forceinline__
void SetArray_WarpSeq
( 
   valT * basePtr,      // IN/OUT - array to set to 'set' value
   valT   toSet,        // IN - value to set array elements 'to'
   uint    startIdx      // starting index for this thread
) 
{
   // Get 'per thread' pointer
   valT * setPtr  = &basePtr[startIdx];

		// Initialize as many elements as we
		// safely can with no range checking
	if (nSafePasses >=  1u) { setPtr[( 0u * WarpSize)] = toSet; }
	if (nSafePasses >=  2u) { setPtr[( 1u * WarpSize)] = toSet; }
	if (nSafePasses >=  3u) { setPtr[( 2u * WarpSize)] = toSet; }
	if (nSafePasses >=  4u) { setPtr[( 3u * WarpSize)] = toSet; }
	if (nSafePasses >=  5u) { setPtr[( 4u * WarpSize)] = toSet; }
	if (nSafePasses >=  6u) { setPtr[( 5u * WarpSize)] = toSet; }
	if (nSafePasses >=  7u) { setPtr[( 6u * WarpSize)] = toSet; }
	if (nSafePasses >=  8u) { setPtr[( 7u * WarpSize)] = toSet; }
	if (nSafePasses >=  9u) { setPtr[( 8u * WarpSize)] = toSet; }
	if (nSafePasses >= 10u) { setPtr[( 9u * WarpSize)] = toSet; }
	if (nSafePasses >= 11u) { setPtr[(10u * WarpSize)] = toSet; }
	if (nSafePasses >= 12u) { setPtr[(11u * WarpSize)] = toSet; }
	if (nSafePasses >= 13u) { setPtr[(12u * WarpSize)] = toSet; }
	if (nSafePasses >= 14u) { setPtr[(13u * WarpSize)] = toSet; }
	if (nSafePasses >= 15u) { setPtr[(14u * WarpSize)] = toSet; }
	if (nSafePasses >= 16u) { setPtr[(15u * WarpSize)] = toSet; }
	if (nSafePasses >= 17u) { setPtr[(16u * WarpSize)] = toSet; }
	if (nSafePasses >= 18u) { setPtr[(17u * WarpSize)] = toSet; }
	if (nSafePasses >= 19u) { setPtr[(18u * WarpSize)] = toSet; }
	if (nSafePasses >= 20u) { setPtr[(19u * WarpSize)] = toSet; }
	if (nSafePasses >= 21u) { setPtr[(20u * WarpSize)] = toSet; }
	if (nSafePasses >= 22u) { setPtr[(21u * WarpSize)] = toSet; }
	if (nSafePasses >= 23u) { setPtr[(22u * WarpSize)] = toSet; }
	if (nSafePasses >= 24u) { setPtr[(23u * WarpSize)] = toSet; }
	if (nSafePasses >= 25u) { setPtr[(24u * WarpSize)] = toSet; }
	if (nSafePasses >= 26u) { setPtr[(25u * WarpSize)] = toSet; }
	if (nSafePasses >= 27u) { setPtr[(26u * WarpSize)] = toSet; }
	if (nSafePasses >= 28u) { setPtr[(27u * WarpSize)] = toSet; }
	if (nSafePasses >= 29u) { setPtr[(28u * WarpSize)] = toSet; }
	if (nSafePasses >= 30u) { setPtr[(29u * WarpSize)] = toSet; }
	if (nSafePasses >= 31u) { setPtr[(30u * WarpSize)] = toSet; }
	if (nSafePasses >= 32u) { setPtr[(31u * WarpSize)] = toSet; }
	if (nSafePasses >= 33u) { setPtr[(32u * WarpSize)] = toSet; }
	if (nSafePasses >= 34u) { setPtr[(33u * WarpSize)] = toSet; }
	if (nSafePasses >= 35u) { setPtr[(34u * WarpSize)] = toSet; }
	if (nSafePasses >= 36u) { setPtr[(35u * WarpSize)] = toSet; }
	if (nSafePasses >= 37u) { setPtr[(36u * WarpSize)] = toSet; }
	if (nSafePasses >= 38u) { setPtr[(37u * WarpSize)] = toSet; }
	if (nSafePasses >= 39u) { setPtr[(38u * WarpSize)] = toSet; }
	if (nSafePasses >= 40u) { setPtr[(39u * WarpSize)] = toSet; }
	if (nSafePasses >= 41u) { setPtr[(40u * WarpSize)] = toSet; }
	if (nSafePasses >= 42u) { setPtr[(41u * WarpSize)] = toSet; }
	if (nSafePasses >= 43u) { setPtr[(42u * WarpSize)] = toSet; }
	if (nSafePasses >= 44u) { setPtr[(43u * WarpSize)] = toSet; }
	if (nSafePasses >= 45u) { setPtr[(44u * WarpSize)] = toSet; }
	if (nSafePasses >= 46u) { setPtr[(45u * WarpSize)] = toSet; }
	if (nSafePasses >= 47u) { setPtr[(46u * WarpSize)] = toSet; }
	if (nSafePasses >= 48u) { setPtr[(47u * WarpSize)] = toSet; }
	if (nSafePasses >= 49u) { setPtr[(48u * WarpSize)] = toSet; }
	if (nSafePasses >= 50u) { setPtr[(49u * WarpSize)] = toSet; }
	if (nSafePasses >= 51u) { setPtr[(50u * WarpSize)] = toSet; }
	if (nSafePasses >= 52u) { setPtr[(51u * WarpSize)] = toSet; }
	if (nSafePasses >= 53u) { setPtr[(52u * WarpSize)] = toSet; }
	if (nSafePasses >= 54u) { setPtr[(53u * WarpSize)] = toSet; }
	if (nSafePasses >= 55u) { setPtr[(54u * WarpSize)] = toSet; }
	if (nSafePasses >= 56u) { setPtr[(55u * WarpSize)] = toSet; }
	if (nSafePasses >= 57u) { setPtr[(56u * WarpSize)] = toSet; }
	if (nSafePasses >= 58u) { setPtr[(57u * WarpSize)] = toSet; }
	if (nSafePasses >= 59u) { setPtr[(58u * WarpSize)] = toSet; }
	if (nSafePasses >= 60u) { setPtr[(59u * WarpSize)] = toSet; }
	if (nSafePasses >= 61u) { setPtr[(60u * WarpSize)] = toSet; }
	if (nSafePasses >= 62u) { setPtr[(61u * WarpSize)] = toSet; }
	if (nSafePasses >= 63u) { setPtr[(62u * WarpSize)] = toSet; }
	if (nSafePasses >= 64u) { setPtr[(63u * WarpSize)] = toSet; }
	if (nSafePasses >= 65u) { setPtr[(64u * WarpSize)] = toSet; }
	if (nSafePasses >= 66u) { setPtr[(65u * WarpSize)] = toSet; }

	// Set any 'left over' values with range checking
	if (nLeftOver > 0u)
	{ 
		uint idx = startIdx + (nSafePasses * WarpSize);
		if (idx < maxSize)
		{
			basePtr[idx] = toSet;
		}
	}
}


/*-------------------------------------------------------------------
  Name:   Bin4_None
  Desc:   *NO* Range check on binning
         Bins 1st & 3rd bytes in value
         Bins 2nd & 4th bytes in value
 ------------------------------------------------------------------*/

template <uint BlockSize>
__device__ __forceinline__
void Bin4_None
( 
   uint * cntPtr, // OUT - count array (to store bin results in)
   uint   val32   // IN  - input 'value' to count
) 
{
   //const uint maskRow13 = 0x003F003Fu;  // Mask for 1st and 3rd lanes
   const uint maskRow13 = 0x0FC00FC0u;
   const uint maskCol   = 0x03030303u;  // Mask for columns

   // Get 'LaneRows' from bins
      // [0..63] = [0..255]/4
   //uint laneRow13 = val32 >> 2u;     // Divide by 4
   //uint laneRow24 = val32 >> 10u;    // Shift by 8, divide by 4

   uint laneRow13 = val32 << 4u;
   uint laneRow24 = val32 >> 4u;

   // Get Lane Column from bins
      // [0..3] = bin [0..255] % 4
   uint laneCol = val32 & maskCol;

      // Mask off 'laneRows' to avoid extra info
   uint LI_13 = laneRow13 & maskRow13;  // get lanes for 1 & 3 bins
   uint LI_24 = laneRow24 & maskRow13;  // get lanes for 2 & 4 bins

   // Get local indices
   //uint LI_13 = laneRow13 * BlockSize;
   //uint LI_24 = laneRow24 * BlockSize;

   // Get Shifts [0,8,16,24] = [0,1,2,3]*8
   uint shift = laneCol << 3u;

   // Get local indices
   uint LI_4 = LI_24 >> 16u;
   uint LI_3 = LI_13 >> 16u;
   uint LI_2 = LI_24 & 0xFFFFu;
   uint LI_1 = LI_13 & 0xFFFFu;

   uint s4 = (shift >> 24u);
   uint s3 = (shift >> 16u);
   uint s2 = (shift >>  8u);
   uint s1 = shift & 0xFFu;

   s3 = s3 & 0xFFu;
   s2 = s2 & 0xFFu;

   uint inc4 = 1u << s4;
   uint inc3 = 1u << s3;
   uint inc2 = 1u << s2;
   uint inc1 = 1u << s1;
   
   uint oldCnt, newCnt;

   // Increment 4th bin
   oldCnt = cntPtr[LI_4];
   newCnt = oldCnt + inc4;
   cntPtr[LI_4] = newCnt;

   // Increment 3rd bin
   oldCnt = cntPtr[LI_3];
   newCnt = oldCnt + inc3;
   cntPtr[LI_3] = newCnt;

   // Increment 2nd bin
   oldCnt = cntPtr[LI_2];
   newCnt = oldCnt + inc2;
   cntPtr[LI_2] = newCnt;

   // Increment 1st bin
   oldCnt = cntPtr[LI_1];
   newCnt = oldCnt + inc1;
   cntPtr[LI_1] = newCnt;
}


/*-------------------------------------------------------------------
  Name:   SS_Sums_4_Next_V1
  Desc:   Serial scan on next 4 elements in seq [0..3]
 ------------------------------------------------------------------*/

template < uint BlockSize,     // Threads per block
           uint BlockMask >    // Block Mask
__device__ __forceinline__
void SS_Sums_4_Next_V1
( 
   uint & sum1,     // OUT - sum1 .. sum4 (as singletons)
   uint & sum2,
   uint & sum3,
   uint & sum4,
   uint * cntPtr,   // IN  - 'per thread' counts <horizontal row> to sum up
   uint   baseIdx
) 
{
   // wrap = (idx + [0..3]) % BlockSize
   uint idx1, idx2, idx3, idx4;
   idx1 = baseIdx + 0u;
   idx2 = baseIdx + 1u;
   idx3 = baseIdx + 2u;
   idx4 = baseIdx + 3u;

   uint wrap1, wrap2, wrap3, wrap4;
   wrap1 = idx1 & BlockMask;
   wrap2 = idx2 & BlockMask;
   wrap3 = idx3 & BlockMask;
   wrap4 = idx4 & BlockMask;

   //-
   // Grab 4 elements in seq [0..3]
   //-

   uint lane1, lane2, lane3, lane4;
   lane1 = cntPtr[wrap1];
   lane2 = cntPtr[wrap2];
   lane3 = cntPtr[wrap3];
   lane4 = cntPtr[wrap4];


   //-
   // Zero out sequence [0..3]
   //-

   cntPtr[wrap1] = 0u;
   cntPtr[wrap2] = 0u;
   cntPtr[wrap3] = 0u;
   cntPtr[wrap4] = 0u;


   //-
   // Accumulate all 4 groups in each lane
   //-

   //-
   // Initialize sums from 1st lane (of 4 groups)
   //-
   uint s3 = lane1 >> 16u;     // 3rd bin (of 4) in lane
   uint s2 = lane1 >>  8u;     // 2nd bin (of 4) in lane

   uint cnt4 = lane1 >> 24u;
   uint cnt3 = s3 & 0xFFu;
   uint cnt2 = s2 & 0xFFu;
   uint cnt1 = lane1 & 0xFFu;

   sum4 = sum4 + cnt4;
   sum3 = sum3 + cnt3;
   sum2 = sum2 + cnt2;
   sum1 = sum1 + cnt1;


   //-
   // Accumulate sums from 2nd lane (of 4 groups)
   //-

   s3 = lane2 >> 16u;     // 3rd bin (of 4) in lane
   s2 = lane2 >>  8u;     // 2nd bin (of 4) in lane

   cnt4 = lane2 >> 24u;
   cnt3 = s3 & 0xFFu;
   cnt2 = s2 & 0xFFu;
   cnt1 = lane2 & 0xFFu;

   sum4 = sum4 + cnt4;
   sum3 = sum3 + cnt3;
   sum2 = sum2 + cnt2;
   sum1 = sum1 + cnt1;


   //-
   // Accumulate sums from 3rd lane (of 4 groups)
   //-

   s3 = lane3 >> 16u;     // 3rd bin (of 4) in lane
   s2 = lane3 >>  8u;     // 2nd bin (of 4) in lane

   cnt4 = lane3 >> 24u;
   cnt3 = s3 & 0xFFu;
   cnt2 = s2 & 0xFFu;
   cnt1 = lane3 & 0xFFu;

   sum4 = sum4 + cnt4;
   sum3 = sum3 + cnt3;
   sum2 = sum2 + cnt2;
   sum1 = sum1 + cnt1;


   //-
   // Accumulate sums from 4th lane (of 4 groups)
   //-

   s3 = lane4 >> 16u;     // 3rd bin (of 4) in lane
   s2 = lane4 >>  8u;     // 2nd bin (of 4) in lane

   cnt4 = lane4 >> 24u;
   cnt3 = s3 & 0xFFu;
   cnt2 = s2 & 0xFFu;
   cnt1 = lane4 & 0xFFu;

   sum4 = sum4 + cnt4;
   sum3 = sum3 + cnt3;
   sum2 = sum2 + cnt2;
   sum1 = sum1 + cnt1;
}


/*-------------------------------------------------------------------
  Name:   SS_Sums_4_Next_V2
  Desc:   Serial scan on next 4 elements in seq [0..3]
 ------------------------------------------------------------------*/

template < uint BlockSize,     // Threads Per Block
           uint BlockMask >    // Block Mask
__device__ __forceinline__
void SS_Sums_4_Next_V2
( 
   uint & sum13,    // OUT - sum1 .. sum4 (as pairs)
   uint & sum24,
   uint * cntPtr,   // IN  - 'per thread' counts <horizontal row> to sum up
   uint   baseIdx
) 
{
   // wrap = (idx + [0..3]) % BlockSize
   uint idx1, idx2, idx3, idx4;
   idx1 = baseIdx + 0u;
   idx2 = baseIdx + 1u;
   idx3 = baseIdx + 2u;
   idx4 = baseIdx + 3u;

   uint wrap1, wrap2, wrap3, wrap4;
   wrap1 = idx1 & BlockMask;
   wrap2 = idx2 & BlockMask;
   wrap3 = idx3 & BlockMask;
   wrap4 = idx4 & BlockMask;

   //-
   // Grab 4 elements in seq [0..3]
   //-

   uint lane1, lane2, lane3, lane4;
   lane1 = cntPtr[wrap1];
   lane2 = cntPtr[wrap2];
   lane3 = cntPtr[wrap3];
   lane4 = cntPtr[wrap4];


   //-
   // Zero out sequence [0..3]
   //-

   cntPtr[wrap1] = 0u;
   cntPtr[wrap2] = 0u;
   cntPtr[wrap3] = 0u;
   cntPtr[wrap4] = 0u;


   //-
   // Accumulate all 4 groups in each lane
   //-

   //-
   // Initialize sums from 1st lane (of 4 groups)
   //-

   uint cnt13, cnt24;
   cnt13 = (lane1 >> 0u) & 0x00FF00FFu;
   cnt24 = (lane1 >> 8u) & 0x00FF00FFu;
   sum13 += cnt13;
   sum24 += cnt24;


   //-
   // Accumulate sums from 2nd lane (of 4 groups)
   //-

   cnt13 = (lane2 >> 0u) & 0x00FF00FFu;
   cnt24 = (lane2 >> 8u) & 0x00FF00FFu;
   sum13 += cnt13;
   sum24 += cnt24;


   //-
   // Accumulate sums from 3rd lane (of 4 groups)
   //-

   cnt13 = (lane3 >> 0u) & 0x00FF00FFu;
   cnt24 = (lane3 >> 8u) & 0x00FF00FFu;
   sum13 += cnt13;
   sum24 += cnt24;


   //-
   // Accumulate sums from 4th lane (of 4 groups)
   //-

   cnt13 = (lane4 >> 0u) & 0x00FF00FFu;
   cnt24 = (lane4 >> 8u) & 0x00FF00FFu;
   sum13 += cnt13;
   sum24 += cnt24;
}


/*-------------------------------------------------------------------
  Name:   AddThreadToRowCounts_V1
  Desc:   Accumulates 'Per Thread' counts into 'Per Row' Counts
 ------------------------------------------------------------------*/

template < uint BlockSize,     // Threads per Block
           uint BlockMask >    // Block Mask
__device__ __forceinline__
void AddThreadToRowCounts_V1
( 
   uint & rCnt1,    // OUT - 4 'per row' counts assigned to this thread
   uint & rCnt2,    //       ditto
   uint & rCnt3,    //       ditto
   uint & rCnt4,    //       ditto
   uint * basePtr,  // IN  - array of 'per thread' counts
   uint   tid
) 
{
   //-----
   // Serial Scan (Scan All 64 elements in sequence)
   //-----

   // Accumulate [0..63]
      // Note: Also zeros out [0..63]
   SS_Sums_4_Next_V1< BlockSize, BlockMask >( rCnt1, rCnt2, rCnt3, rCnt4, basePtr, (tid +  0) );
   SS_Sums_4_Next_V1< BlockSize, BlockMask >( rCnt1, rCnt2, rCnt3, rCnt4, basePtr, (tid +  4) );
   SS_Sums_4_Next_V1< BlockSize, BlockMask >( rCnt1, rCnt2, rCnt3, rCnt4, basePtr, (tid +  8) );
   SS_Sums_4_Next_V1< BlockSize, BlockMask >( rCnt1, rCnt2, rCnt3, rCnt4, basePtr, (tid + 12) );
   SS_Sums_4_Next_V1< BlockSize, BlockMask >( rCnt1, rCnt2, rCnt3, rCnt4, basePtr, (tid + 16) );
   SS_Sums_4_Next_V1< BlockSize, BlockMask >( rCnt1, rCnt2, rCnt3, rCnt4, basePtr, (tid + 20) );
   SS_Sums_4_Next_V1< BlockSize, BlockMask >( rCnt1, rCnt2, rCnt3, rCnt4, basePtr, (tid + 24) );
   SS_Sums_4_Next_V1< BlockSize, BlockMask >( rCnt1, rCnt2, rCnt3, rCnt4, basePtr, (tid + 28) );
   SS_Sums_4_Next_V1< BlockSize, BlockMask >( rCnt1, rCnt2, rCnt3, rCnt4, basePtr, (tid + 32) );
   SS_Sums_4_Next_V1< BlockSize, BlockMask >( rCnt1, rCnt2, rCnt3, rCnt4, basePtr, (tid + 36) );
   SS_Sums_4_Next_V1< BlockSize, BlockMask >( rCnt1, rCnt2, rCnt3, rCnt4, basePtr, (tid + 40) );
   SS_Sums_4_Next_V1< BlockSize, BlockMask >( rCnt1, rCnt2, rCnt3, rCnt4, basePtr, (tid + 44) );
   SS_Sums_4_Next_V1< BlockSize, BlockMask >( rCnt1, rCnt2, rCnt3, rCnt4, basePtr, (tid + 48) );
   SS_Sums_4_Next_V1< BlockSize, BlockMask >( rCnt1, rCnt2, rCnt3, rCnt4, basePtr, (tid + 52) );
   SS_Sums_4_Next_V1< BlockSize, BlockMask >( rCnt1, rCnt2, rCnt3, rCnt4, basePtr, (tid + 56) );
   SS_Sums_4_Next_V1< BlockSize, BlockMask >( rCnt1, rCnt2, rCnt3, rCnt4, basePtr, (tid + 60) );
}


/*-------------------------------------------------------------------
  Name:   AddThreadToRowCounts_V2
  Desc:   Accumulates 'Per Thread' counts into 'Per Row' Counts
  Notes:   
  1. Vector Parallelism: 
       We accumulate 2 pairs at a time across each row 
       instead of 4 singletons for a big savings 
       in arithmetic operations.
  2. Overflow:
       We store 2 16-bit row sums per 32-bit number
       Which means that the accumulated Row sums need to not
       overflow a 16-bit number (65,535). 
       Since, we assume the maximum possible count per thread is 252
          64 threads * 252 =  16,128 <Safe>
         128 threads * 252 =  32,256 <Safe>
         256 threads * 252 =  64,512 <Safe>
         512 threads * 252 = 129,024 *** UNSAFE ***
       If this is a problem, revert to *_V1
  3. Register Pressure:
       *_V2 uses 6 more registers per thread than *_V1
       If this is a problem, revert to *_V1
 ------------------------------------------------------------------*/

template < uint BlockSize,     // Threads per Block
           uint BlockMask >    // BlockSize - 1
__device__ __forceinline__
void AddThreadToRowCounts_V2
( 
   uint & rCnt1,    // OUT - 4 'per row' counts assigned to this thread
   uint & rCnt2,    //       ditto
   uint & rCnt3,    //       ditto
   uint & rCnt4,    //       ditto
   uint * basePtr,  // IN  - array of 'per thread' counts
   uint   tid       // IN  - thread ID
) 
{
   uint sum13, sum24;
   sum13 = 0u;
   sum24 = 0u;

   //-----
   // Serial Scan (Scan All 64 elements in sequence)
   //-----

   // Accumulate Row Sums [0..63]
      // Note: Also zeros out count array while accumulating
   SS_Sums_4_Next_V2< BlockSize, BlockMask >( sum13, sum24, basePtr, (tid +  0) );
   SS_Sums_4_Next_V2< BlockSize, BlockMask >( sum13, sum24, basePtr, (tid +  4) );
   SS_Sums_4_Next_V2< BlockSize, BlockMask >( sum13, sum24, basePtr, (tid +  8) );
   SS_Sums_4_Next_V2< BlockSize, BlockMask >( sum13, sum24, basePtr, (tid + 12) );
   SS_Sums_4_Next_V2< BlockSize, BlockMask >( sum13, sum24, basePtr, (tid + 16) );
   SS_Sums_4_Next_V2< BlockSize, BlockMask >( sum13, sum24, basePtr, (tid + 20) );
   SS_Sums_4_Next_V2< BlockSize, BlockMask >( sum13, sum24, basePtr, (tid + 24) );
   SS_Sums_4_Next_V2< BlockSize, BlockMask >( sum13, sum24, basePtr, (tid + 28) );
   SS_Sums_4_Next_V2< BlockSize, BlockMask >( sum13, sum24, basePtr, (tid + 32) );
   SS_Sums_4_Next_V2< BlockSize, BlockMask >( sum13, sum24, basePtr, (tid + 36) );
   SS_Sums_4_Next_V2< BlockSize, BlockMask >( sum13, sum24, basePtr, (tid + 40) );
   SS_Sums_4_Next_V2< BlockSize, BlockMask >( sum13, sum24, basePtr, (tid + 44) );
   SS_Sums_4_Next_V2< BlockSize, BlockMask >( sum13, sum24, basePtr, (tid + 48) );
   SS_Sums_4_Next_V2< BlockSize, BlockMask >( sum13, sum24, basePtr, (tid + 52) );
   SS_Sums_4_Next_V2< BlockSize, BlockMask >( sum13, sum24, basePtr, (tid + 56) );
   SS_Sums_4_Next_V2< BlockSize, BlockMask >( sum13, sum24, basePtr, (tid + 60) );

   // Convert row sums from pairs back into singletons
   uint sum1, sum2, sum3, sum4;
   sum1 = sum13 & 0x0000FFFFu;
   sum2 = sum24 & 0x0000FFFFu;
   sum3 = sum13 >> 16u;
   sum4 = sum24 >> 16u;

   // Add row sums back into register counts
   rCnt1 += sum1;
   rCnt2 += sum2;
   rCnt3 += sum3;
   rCnt4 += sum4;
}


/*---------------------------------------------------------
  Name:   H_K1_CountRows_256_TRISH
  Desc:   Bins elements into 256-way row counts
 ---------------------------------------------------------*/

template < uint logBankSize,   // log<2>( Channels per Bank )
           uint logWarpSize,	  // log<2>( Threads per Warp )
           uint BlockSize,     // Threads Per Block (needs to be a power of 2 & multiple of warpsize)
		   uint GridSize,	  // Blocks Per Grid
           uint K_length >     // #elements to process per thread before looping
__global__
void H_K1_CountRows_256_TRISH
( 
         uint * outRowCounts,  // OUT - 256-way row-sums array
   const uint * inVals,		   // IN  - values to bin and count
         uint   start,         // IN  - range [start,stop] to check and count
         uint   stop           //       ditto
) 
{
	//-------------------------------------------
	// Constant values (computed at compile time)
	//-------------------------------------------

		// Bank Size (elements per bank)
	const uint BankSize    = (1u << logBankSize);	   // 32 = 2^5 threads per bank
	const uint BankMask    = BankSize - 1u;	         // 31 = 32 - 1 = 0x1F = b11111
   const uint strideBank  = BankSize + 1u;          // 33 = 32 + 1
      // Extra '+1' to help try and avoid bank conflicts

		// Warp Size (threads per warp)
	const uint WarpSize    = (1u << logWarpSize);	   // 32 = 2^5 threads per warp
	const uint WarpMask    = WarpSize - 1u;			   // 31 = 32 - 1 = 0x1F = b11111

      // Block Size (threads per block)
   //const uint BlockSize   = 64u;
   const uint BlockMask   = BlockSize - 1u;

		// Chunk Size
	//const uint ChunkSize     = BlockSize * K_length;
   //const uint IN_WarpSize   = K_length * WarpSize;

      // K_length
   //const uint K_length = 16u;               // 16 
   const uint K4_length = K_length * 4u;      // 64 = 16 * 4
   const uint K4_stop   = 256u - K4_length;   // 192 = 256 - 64

		// Warps Per Block
	const uint WarpsPerBlock = BlockSize / WarpSize;   // 2 = 64/32

		// Bins per Histogram
	const uint nHistBins     = 256u;     // 256 = 2^8

		// Lane Info (Compress 4 'bins' into each 32-bit value)
	const uint nLanes		   = 64u;   // 64, # Lanes = 256 bins / 4 bins per lane

		// 'Per Thread' counts array
	const uint nTCounts      = nLanes * BlockSize;
	const uint banksTCounts  = (nTCounts + BankMask) / BankSize;
	const uint padTCounts    = (banksTCounts * BankSize) - nTCounts;
	const uint sizeTCounts   = nTCounts + padTCounts;

      // Output size
   const uint OutWarpSize   = nHistBins / WarpsPerBlock;
   const uint OutLength     = OutWarpSize / WarpSize;
   const uint OutStrideSize = OutLength * strideBank;

		// Array Initialization
	const uint nPassesThrd  = sizeTCounts / BlockSize;
	const uint leftOverThrd = sizeTCounts - (nPassesThrd * BlockSize);

	const uint nThreadsPerGrid = BlockSize * GridSize;	//   3,072 = 64 * 48
   const uint rowSize = K_length * nThreadsPerGrid;		// 193,586 = 63 * 64 * 48


	//------------------------------------
	// Local Variables
	//------------------------------------

		// Local variables (shared memory)
	__shared__ uint  s_thrdCounts[sizeTCounts];   // 'per thread' counts

      // Local variables (registers)
   uint rowCnt1 = 0u;
   uint rowCnt2 = 0u;
   uint rowCnt3 = 0u; 
   uint rowCnt4 = 0u;

	//---------------------------
	// Compute Indices & Pointers
	//---------------------------

   uint tid = threadIdx.x;		// Thread ID within Block
   uint * cntPtr;
   uint * basePtr;

   {
      // Get Warp Row & Column
      //uint warpRow = threadIdx.x >> logWarpSize; // tid / 32
      //uint warpCol = threadIdx.x & WarpMask;     // tid % 32

      // Compute starting 'input' offset (Warp Sequential Layout)
      //inIdx = (warpRow * IN_WarpSize) // Move to each warps assigned portion of work
      //        + warpCol;              // Move to warp column (in warp)

         // Compute starting serial scan index
      uint baseIdx = (tid * BlockSize);

         // Get pointers into shared memory array
         // for different views of memory
      cntPtr  = &s_thrdCounts[threadIdx.x];
      basePtr = &s_thrdCounts[baseIdx];
   }


	//-------------------------------------------
	// Zero out arrays
	//-------------------------------------------

   {
	   //-
	   // Zero out 'Per Thread' counts
	   //-

      uint * ptrTC = (&s_thrdCounts[0]);
      SetArray_BlockSeq
         < 
            uint, BlockSize, nPassesThrd, leftOverThrd, sizeTCounts
         >
         ( 
            ptrTC, 0u
         );
   }

	// Sync Threads in Block
	if (WarpsPerBlock >= 2u) { __syncthreads(); }

   //-----
   // Compute thread, block, & grid indices & sizes
   //-----
 
   uint bid = (blockIdx.y * gridDim.x) + blockIdx.x;		// Block ID within Grid
   uint elemOffset = (bid * K_length * BlockSize) + tid;	// Starting offset 

   uint nElems32        = stop - start + 1u;
   uint nMaxRows        = (nElems32 + (rowSize - 1u)) / rowSize;
   uint nSafeRows       = nElems32 / rowSize;
   uint nSafeElems      = nSafeRows * rowSize;
   uint nLeftOverElems  = nElems32 - nSafeElems;

   uint startIdx        = start + elemOffset;
   uint stopIdx         = startIdx + (nSafeRows * rowSize);
   uint currIdx         = startIdx;
   uint overflow        = 0u;


   //-----
   // Process all safe blocks
   //-----

   // 'input' pointer for reading from memory
   const uint * inPtr = &inVals[currIdx];

   while (currIdx < stopIdx)
	{
      //-----
      // Accumulate 'thread' counts into 'row' counts
      //    Note: Also zeros out 'per thread' count array
      //-----

      if (overflow >= K4_stop)
      {
         // Sync Threads in Block
         if (WarpsPerBlock >= 2u) { __syncthreads(); }

         //AddThreadToRowCounts_V1< BlockSize, BlockMask >( rowCnt1, rowCnt2, rowCnt3, rowCnt4, basePtr, tid );
         AddThreadToRowCounts_V2< BlockSize, BlockMask >( rowCnt1, rowCnt2, rowCnt3, rowCnt4, basePtr, tid );
         overflow = 0u;

         // Sync Threads in Block
         if (WarpsPerBlock >= 2u) { __syncthreads(); }
      }

      uint val1, val2, val3, val4;

         // NOTE:  the 'K_length' variable below is a static
         //        hard-coded constant in the range [1..63].
         //        K = 'Work per thread' per loop (stride)...
         //        The compiler will take care of throwing away 
         //        any unused code greater than our specified 'K'
         //        value, with no negative impact on performance.

      //-
      // Process values [0..3] (bytes 0..15)
      //-

      // Read in first 'four' values (32-bit)
      if (K_length >= 1u) { val1 = inPtr[0u*BlockSize]; }
      if (K_length >= 2u) { val2 = inPtr[1u*BlockSize]; }
      if (K_length >= 3u) { val3 = inPtr[2u*BlockSize]; }
      if (K_length >= 4u) { val4 = inPtr[3u*BlockSize]; }

      // Bin first 'four' values
      if (K_length >= 1u) { Bin4_None<BlockSize>( cntPtr, val1 ); }
      if (K_length >= 2u) { Bin4_None<BlockSize>( cntPtr, val2 ); }
      if (K_length >= 3u) { Bin4_None<BlockSize>( cntPtr, val3 ); }
      if (K_length >= 4u) { Bin4_None<BlockSize>( cntPtr, val4 ); }


      //-
      // Process values [4..7] (bytes 16..31)
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 5u) { val1 = inPtr[4u*BlockSize]; }
      if (K_length >= 6u) { val2 = inPtr[5u*BlockSize]; }
      if (K_length >= 7u) { val3 = inPtr[6u*BlockSize]; }
      if (K_length >= 8u) { val4 = inPtr[7u*BlockSize]; }

      // Bin 'four' values (4 bytes at a time)
      if (K_length >= 5u) { Bin4_None<BlockSize>( cntPtr, val1 ); }
      if (K_length >= 6u) { Bin4_None<BlockSize>( cntPtr, val2 ); }
      if (K_length >= 7u) { Bin4_None<BlockSize>( cntPtr, val3 ); }
      if (K_length >= 8u) { Bin4_None<BlockSize>( cntPtr, val4 ); }


      //-
      // Process values [8..11] (bytes 32..47)
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >=  9u) { val1 = inPtr[ 8u*BlockSize]; } 
      if (K_length >= 10u) { val2 = inPtr[ 9u*BlockSize]; }
      if (K_length >= 11u) { val3 = inPtr[10u*BlockSize]; }
      if (K_length >= 12u) { val4 = inPtr[11u*BlockSize]; }

      // Bin 'four' values (4 bytes at a time)
      if (K_length >=  9u) { Bin4_None<BlockSize>( cntPtr, val1 ); }
      if (K_length >= 10u) { Bin4_None<BlockSize>( cntPtr, val2 ); }
      if (K_length >= 11u) { Bin4_None<BlockSize>( cntPtr, val3 ); }
      if (K_length >= 12u) { Bin4_None<BlockSize>( cntPtr, val4 ); }

      //-
      // Process values [12..15] (bytes 48..63)
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 13u) { val1 = inPtr[12u*BlockSize]; }
      if (K_length >= 14u) { val2 = inPtr[13u*BlockSize]; }
      if (K_length >= 15u) { val3 = inPtr[14u*BlockSize]; }
      if (K_length >= 16u) { val4 = inPtr[15u*BlockSize]; }

      // Bin 'four' values (4 bytes at a time)
      if (K_length >= 13u) { Bin4_None<BlockSize>( cntPtr, val1 ); }
      if (K_length >= 14u) { Bin4_None<BlockSize>( cntPtr, val2 ); }
      if (K_length >= 15u) { Bin4_None<BlockSize>( cntPtr, val3 ); }
      if (K_length >= 16u) { Bin4_None<BlockSize>( cntPtr, val4 ); }


      //-
      // Process values [16..19] (bytes 64..79)
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 17u) { val1 = inPtr[16u*BlockSize]; }
      if (K_length >= 18u) { val2 = inPtr[17u*BlockSize]; }
      if (K_length >= 19u) { val3 = inPtr[18u*BlockSize]; }
      if (K_length >= 20u) { val4 = inPtr[19u*BlockSize]; }

      // Bin 'four' values (4 bytes at a time)
      if (K_length >= 17u) { Bin4_None<BlockSize>( cntPtr, val1 ); }
      if (K_length >= 18u) { Bin4_None<BlockSize>( cntPtr, val2 ); }
      if (K_length >= 19u) { Bin4_None<BlockSize>( cntPtr, val3 ); }
      if (K_length >= 20u) { Bin4_None<BlockSize>( cntPtr, val4 ); }


      //-
      // Process values [20..23] (bytes 80..95)
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 21u) { val1 = inPtr[20u*BlockSize]; }
      if (K_length >= 22u) { val2 = inPtr[21u*BlockSize]; }
      if (K_length >= 23u) { val3 = inPtr[22u*BlockSize]; }
      if (K_length >= 24u) { val4 = inPtr[23u*BlockSize]; }

      // Bin 'four' values (4 bytes at a time)
      if (K_length >= 21u) { Bin4_None<BlockSize>( cntPtr, val1 ); }
      if (K_length >= 22u) { Bin4_None<BlockSize>( cntPtr, val2 ); }
      if (K_length >= 23u) { Bin4_None<BlockSize>( cntPtr, val3 ); }
      if (K_length >= 24u) { Bin4_None<BlockSize>( cntPtr, val4 ); }


      //-
      // Process values [24..27] (bytes 96..111)
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 25u) { val1 = inPtr[24u*BlockSize]; }
      if (K_length >= 26u) { val2 = inPtr[25u*BlockSize]; }
      if (K_length >= 27u) { val3 = inPtr[26u*BlockSize]; }
      if (K_length >= 28u) { val4 = inPtr[27u*BlockSize]; }

      // Bin 'four' values (4 bytes at a time)
      if (K_length >= 25u) { Bin4_None<BlockSize>( cntPtr, val1 ); }
      if (K_length >= 26u) { Bin4_None<BlockSize>( cntPtr, val2 ); }
      if (K_length >= 27u) { Bin4_None<BlockSize>( cntPtr, val3 ); }
      if (K_length >= 28u) { Bin4_None<BlockSize>( cntPtr, val4 ); }


      //-
      // Process values [28..31] (bytes 112..127)
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 29u) { val1 = inPtr[28u*BlockSize]; }
      if (K_length >= 30u) { val2 = inPtr[29u*BlockSize]; }
      if (K_length >= 31u) { val3 = inPtr[30u*BlockSize]; }
      if (K_length >= 32u) { val4 = inPtr[31u*BlockSize]; }

      // Bin 'four' values (4 bytes at a time)
      if (K_length >= 29u) { Bin4_None<BlockSize>( cntPtr, val1 ); }
      if (K_length >= 30u) { Bin4_None<BlockSize>( cntPtr, val2 ); }
      if (K_length >= 31u) { Bin4_None<BlockSize>( cntPtr, val3 ); }
      if (K_length >= 32u) { Bin4_None<BlockSize>( cntPtr, val4 ); }


      //-
      // Process values [32..35] (bytes 128..143)
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 33u) { val1 = inPtr[32u*BlockSize]; }
      if (K_length >= 34u) { val2 = inPtr[33u*BlockSize]; }
      if (K_length >= 35u) { val3 = inPtr[34u*BlockSize]; }
      if (K_length >= 36u) { val4 = inPtr[35u*BlockSize]; }

      // Bin 'four' values (4 bytes at a time)
      if (K_length >= 33u) { Bin4_None<BlockSize>( cntPtr, val1 ); }
      if (K_length >= 34u) { Bin4_None<BlockSize>( cntPtr, val2 ); }
      if (K_length >= 35u) { Bin4_None<BlockSize>( cntPtr, val3 ); }
      if (K_length >= 36u) { Bin4_None<BlockSize>( cntPtr, val4 ); }


      //-
      // Process values [36..39] (bytes 144..159)
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 37u) { val1 = inPtr[36u*BlockSize]; }
      if (K_length >= 38u) { val2 = inPtr[37u*BlockSize]; }
      if (K_length >= 39u) { val3 = inPtr[38u*BlockSize]; }
      if (K_length >= 40u) { val4 = inPtr[39u*BlockSize]; }

      // Bin 'four' values (4 bytes at a time)
      if (K_length >= 37u) { Bin4_None<BlockSize>( cntPtr, val1 ); }
      if (K_length >= 38u) { Bin4_None<BlockSize>( cntPtr, val2 ); }
      if (K_length >= 39u) { Bin4_None<BlockSize>( cntPtr, val3 ); }
      if (K_length >= 40u) { Bin4_None<BlockSize>( cntPtr, val4 ); }

      //-
      // Process values [40..43] (bytes 160-175)
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 41u) { val1 = inPtr[40u*BlockSize]; }
      if (K_length >= 42u) { val2 = inPtr[41u*BlockSize]; }
      if (K_length >= 43u) { val3 = inPtr[42u*BlockSize]; }
      if (K_length >= 44u) { val4 = inPtr[43u*BlockSize]; }

      // Bin 'four' values (4 bytes at a time)
      if (K_length >= 41u) { Bin4_None<BlockSize>( cntPtr, val1 ); }
      if (K_length >= 42u) { Bin4_None<BlockSize>( cntPtr, val2 ); }
      if (K_length >= 43u) { Bin4_None<BlockSize>( cntPtr, val3 ); }
      if (K_length >= 44u) { Bin4_None<BlockSize>( cntPtr, val4 ); }


      //-
      // Process values [44..47] (bytes 176-191)
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 45u) { val1 = inPtr[44u*BlockSize]; }
      if (K_length >= 46u) { val2 = inPtr[45u*BlockSize]; }
      if (K_length >= 47u) { val3 = inPtr[46u*BlockSize]; }
      if (K_length >= 48u) { val4 = inPtr[47u*BlockSize]; }

      // Bin 'four' values (4 bytes at a time)
      if (K_length >= 45u) { Bin4_None<BlockSize>( cntPtr, val1 ); }
      if (K_length >= 46u) { Bin4_None<BlockSize>( cntPtr, val2 ); }
      if (K_length >= 47u) { Bin4_None<BlockSize>( cntPtr, val3 ); }
      if (K_length >= 48u) { Bin4_None<BlockSize>( cntPtr, val4 ); }


      //-
      // Process values [48-51] (bytes 192-207)
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 49u) { val1 = inPtr[48u*BlockSize]; }
      if (K_length >= 50u) { val2 = inPtr[49u*BlockSize]; }
      if (K_length >= 51u) { val3 = inPtr[50u*BlockSize]; }
      if (K_length >= 52u) { val4 = inPtr[51u*BlockSize]; }

      // Bin 'four' values (4 bytes at a time)
      if (K_length >= 49u) { Bin4_None<BlockSize>( cntPtr, val1 ); }
      if (K_length >= 50u) { Bin4_None<BlockSize>( cntPtr, val2 ); }
      if (K_length >= 51u) { Bin4_None<BlockSize>( cntPtr, val3 ); }
      if (K_length >= 52u) { Bin4_None<BlockSize>( cntPtr, val4 ); }


      //-
      // Process values [52-55] (bytes 208-223)
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 53u) { val1 = inPtr[52u*BlockSize]; }
      if (K_length >= 54u) { val2 = inPtr[53u*BlockSize]; }
      if (K_length >= 55u) { val3 = inPtr[54u*BlockSize]; }
      if (K_length >= 56u) { val4 = inPtr[55u*BlockSize]; }

      // Bin 'four' values (4 bytes at a time)
      if (K_length >= 53u) { Bin4_None<BlockSize>( cntPtr, val1 ); }
      if (K_length >= 54u) { Bin4_None<BlockSize>( cntPtr, val2 ); }
      if (K_length >= 55u) { Bin4_None<BlockSize>( cntPtr, val3 ); }
      if (K_length >= 56u) { Bin4_None<BlockSize>( cntPtr, val4 ); }


      //-
      // Process values [56-59] (bytes 224-239)
      //-

      // Read in next 'four' values (32-bit)
      if (K_length >= 57u) { val1 = inPtr[56u*BlockSize]; }
      if (K_length >= 58u) { val2 = inPtr[57u*BlockSize]; }
      if (K_length >= 59u) { val3 = inPtr[58u*BlockSize]; }
      if (K_length >= 60u) { val4 = inPtr[59u*BlockSize]; }

      // Bin 'four' values (4 bytes at a time)
      if (K_length >= 57u) { Bin4_None<BlockSize>( cntPtr, val1 ); }
      if (K_length >= 58u) { Bin4_None<BlockSize>( cntPtr, val2 ); }
      if (K_length >= 59u) { Bin4_None<BlockSize>( cntPtr, val3 ); }
      if (K_length >= 60u) { Bin4_None<BlockSize>( cntPtr, val4 ); }


      //-
      // Process values [60-62] (bytes 240-251)
      //-
         // Note: We deliberately do not support k >= '64' to
         //       avoid overflow issues during 'binning'
         //       As our 'per thread' 'bin counts' can only handle 
         //       '255' increments before overflow becomes a problem.
         //       and 252 is the next smallest number 
         //       evenly divisible by 4, IE 4 bytes per 32-bit value
         //       63 values = 252 bytes / 4 bytes per value.

      // Read in next 'four' values (32-bit)
      if (K_length >= 61u) { val1 = inPtr[60u*BlockSize]; }
      if (K_length >= 62u) { val2 = inPtr[61u*BlockSize]; }
      if (K_length >= 63u) { val3 = inPtr[62u*BlockSize]; }

      // Note: Do not uncomment => *OVERFLOW* bug !!!
      //if (K_length >= 64u) { val4 = inPtr[63u*BlockSize]; }

      // Bin 'four' values (4 bytes at a time)
      if (K_length >= 61u) { Bin4_None<BlockSize>( cntPtr, val1 ); }
      if (K_length >= 62u) { Bin4_None<BlockSize>( cntPtr, val2 ); }
      if (K_length >= 63u) { Bin4_None<BlockSize>( cntPtr, val3 ); }

      // Note: Do not uncomment => *OVERFLOW* bug !!!
      //if (K_length >= 64u) { Bin4_None<BlockSize>( cntPtr, val4 ); }


      // Increment 'overflow' count
      overflow += K4_length;   // K values * 4 bytes per value


      //-----
		// Move to next row of work
		//-----

		currIdx += rowSize;
        inPtr += rowSize;
	}


	//--------------------------------------
	// LAST: Process last leftover chunk
    //       with more careful range checking
	//--------------------------------------

	if (nLeftOverElems)
	{
      //-----
      // Accumulate 'thread' counts into 'row' counts
      //    Note: Also zeros out 'per thread' count array
      //-----

      if (overflow >= K4_stop)
      {
         // Sync Threads in Block
         if (WarpsPerBlock >= 2u) { __syncthreads(); }

         //AddThreadToRowCounts_V1< BlockSize, BlockMask >( rowCnt1, rowCnt2, rowCnt3, rowCnt4, basePtr, tid );
         AddThreadToRowCounts_V2< BlockSize, BlockMask >( rowCnt1, rowCnt2, rowCnt3, rowCnt4, basePtr, tid );
         overflow = 0u;

         // Sync Threads in Block
         if (WarpsPerBlock >= 2u) { __syncthreads(); }
      }

         // NOTE #1:  the 'K_length' variable below is a static
         //           hard-coded constant in the range [1..63].
         //           K = 'Work per thread' per loop (stride)...
         //           The compiler will take care of throwing away 
         //           any unused code greater than our specified 'K'
         //           value, with no negative impact on performance.

         // NOTE #2:  We use a cooperative stride 
         //           across each thread in each block in grid
         //           ChunkSize = BlockSize * GridSize = 64 * 48 = 3072
         //           RowSize   = WorkPerThead(K) * ChunkSize = 63 * 3072 = 193,536
         // 
         //                       B0   B1  ...  B47  (Blocks in Grid)
         //                      ---- ---- --- ----
         //           k =  1 =>  |64| |64| ... |64|  (3072 Thread & I/O requests for 1st work item per thread)
         //           k =  2 =>  |64| |64| ... |64|  ditto (2nd work item per thread)
         //               ...       ...         ...
         //           k = 63 =>  |64| |64| ... |64|  ditto (63 work item per thread)

         // NOTE #3:  We use "Divide & Conquer" to avoid as much slower range checking as possible
         //			  Try batches of 32, 16, 8, 4, 2, 1, and finally leftover (on which we finally must range check) 

      //----
      // Setup Pointers & Indices for cooperative stride 
      //----

      uint bid        = (blockIdx.y * gridDim.x) + blockIdx.x;	// Get block index
      uint nSkip      = nSafeRows * rowSize;						// Skip past already processed rows
      uint chunkIdx   = (bid * BlockSize) + tid;					// Get starting index within chunk
      uint baseIdx    = start + nSkip + chunkIdx;				// Get starting index for left over elements

      uint val1, val2, val3, val4;

      //------
      // Try Section of 32
      //------

      if (K_length >= 32u)
      {
         // Process 32 chunks safely without range checking
         if (nLeftOverElems >= (32u * nThreadsPerGrid))
         {
            // Get pointer
            inPtr = &inVals[baseIdx];

            //-----
            // Read & Bin [1..4]
            //-----

            val1 = inPtr[(0u*nThreadsPerGrid)];
            val2 = inPtr[(1u*nThreadsPerGrid)];
            val3 = inPtr[(2u*nThreadsPerGrid)];
            val4 = inPtr[(3u*nThreadsPerGrid)];

            Bin4_None<BlockSize>( cntPtr, val1 );
            Bin4_None<BlockSize>( cntPtr, val2 );
            Bin4_None<BlockSize>( cntPtr, val3 );
            Bin4_None<BlockSize>( cntPtr, val4 );


            //-----
            // Read & Bin [5..8]
            //-----

            val1 = inPtr[(4u*nThreadsPerGrid)];
            val2 = inPtr[(5u*nThreadsPerGrid)];
            val3 = inPtr[(6u*nThreadsPerGrid)];
            val4 = inPtr[(7u*nThreadsPerGrid)];

            Bin4_None<BlockSize>( cntPtr, val1 );
            Bin4_None<BlockSize>( cntPtr, val2 );
            Bin4_None<BlockSize>( cntPtr, val3 );
            Bin4_None<BlockSize>( cntPtr, val4 );


            //-----
            // Read & Bin [9..12]
            //-----

            val1 = inPtr[( 8u*nThreadsPerGrid)];
            val2 = inPtr[( 9u*nThreadsPerGrid)];
            val3 = inPtr[(10u*nThreadsPerGrid)];
            val4 = inPtr[(11u*nThreadsPerGrid)];

            Bin4_None<BlockSize>( cntPtr, val1 );
            Bin4_None<BlockSize>( cntPtr, val2 );
            Bin4_None<BlockSize>( cntPtr, val3 );
            Bin4_None<BlockSize>( cntPtr, val4 );


            //-----
            // Read & Bin [13..16]
            //-----

            val1 = inPtr[(12u*nThreadsPerGrid)];
            val2 = inPtr[(13u*nThreadsPerGrid)];
            val3 = inPtr[(14u*nThreadsPerGrid)];
            val4 = inPtr[(15u*nThreadsPerGrid)];

            Bin4_None<BlockSize>( cntPtr, val1 );
            Bin4_None<BlockSize>( cntPtr, val2 );
            Bin4_None<BlockSize>( cntPtr, val3 );
            Bin4_None<BlockSize>( cntPtr, val4 );


            //-----
            // Read & Bin [17..20]
            //-----

            val1 = inPtr[(16u*nThreadsPerGrid)];
            val2 = inPtr[(17u*nThreadsPerGrid)];
            val3 = inPtr[(18u*nThreadsPerGrid)];
            val4 = inPtr[(19u*nThreadsPerGrid)];

            Bin4_None<BlockSize>( cntPtr, val1 );
            Bin4_None<BlockSize>( cntPtr, val2 );
            Bin4_None<BlockSize>( cntPtr, val3 );
            Bin4_None<BlockSize>( cntPtr, val4 );


            //-----
            // Read & Bin [21..24]
            //-----

            val1 = inPtr[(20u*nThreadsPerGrid)];
            val2 = inPtr[(21u*nThreadsPerGrid)];
            val3 = inPtr[(22u*nThreadsPerGrid)];
            val4 = inPtr[(23u*nThreadsPerGrid)];

            Bin4_None<BlockSize>( cntPtr, val1 );
            Bin4_None<BlockSize>( cntPtr, val2 );
            Bin4_None<BlockSize>( cntPtr, val3 );
            Bin4_None<BlockSize>( cntPtr, val4 );


            //-----
            // Read & Bin [25..28]
            //-----

            val1 = inPtr[(24u*nThreadsPerGrid)];
            val2 = inPtr[(25u*nThreadsPerGrid)];
            val3 = inPtr[(26u*nThreadsPerGrid)];
            val4 = inPtr[(27u*nThreadsPerGrid)];

            Bin4_None<BlockSize>( cntPtr, val1 );
            Bin4_None<BlockSize>( cntPtr, val2 );
            Bin4_None<BlockSize>( cntPtr, val3 );
            Bin4_None<BlockSize>( cntPtr, val4 );


            //-----
            // Read & Bin [29..32]
            //-----

            val1 = inPtr[(28u*nThreadsPerGrid)];
            val2 = inPtr[(29u*nThreadsPerGrid)];
            val3 = inPtr[(30u*nThreadsPerGrid)];
            val4 = inPtr[(31u*nThreadsPerGrid)];

            Bin4_None<BlockSize>( cntPtr, val1 );
            Bin4_None<BlockSize>( cntPtr, val2 );
            Bin4_None<BlockSize>( cntPtr, val3 );
            Bin4_None<BlockSize>( cntPtr, val4 );


            // Move to next section
            baseIdx        += (32u * nThreadsPerGrid);
            nLeftOverElems -= (32u * nThreadsPerGrid);
         }
      }


      //------
      // Try Section of 16
      //------

      if (K_length >= 16u)
      {
         // Process 16 chunks safely without range checking
         if (nLeftOverElems >= (16u * nThreadsPerGrid))
         {
            // Get pointer
            inPtr = &inVals[baseIdx];

            //-----
            // Read & Bin [1..4]
            //-----

            val1 = inPtr[(0u*nThreadsPerGrid)];
            val2 = inPtr[(1u*nThreadsPerGrid)];
            val3 = inPtr[(2u*nThreadsPerGrid)];
            val4 = inPtr[(3u*nThreadsPerGrid)];

            Bin4_None<BlockSize>( cntPtr, val1 );
            Bin4_None<BlockSize>( cntPtr, val2 );
            Bin4_None<BlockSize>( cntPtr, val3 );
            Bin4_None<BlockSize>( cntPtr, val4 );


            //-----
            // Read & Bin [5..8]
            //-----

            val1 = inPtr[(4u*nThreadsPerGrid)];
            val2 = inPtr[(5u*nThreadsPerGrid)];
            val3 = inPtr[(6u*nThreadsPerGrid)];
            val4 = inPtr[(7u*nThreadsPerGrid)];

            Bin4_None<BlockSize>( cntPtr, val1 );
            Bin4_None<BlockSize>( cntPtr, val2 );
            Bin4_None<BlockSize>( cntPtr, val3 );
            Bin4_None<BlockSize>( cntPtr, val4 );


            //-----
            // Read & Bin [9..12]
            //-----

            val1 = inPtr[( 8u*nThreadsPerGrid)];
            val2 = inPtr[( 9u*nThreadsPerGrid)];
            val3 = inPtr[(10u*nThreadsPerGrid)];
            val4 = inPtr[(11u*nThreadsPerGrid)];

            Bin4_None<BlockSize>( cntPtr, val1 );
            Bin4_None<BlockSize>( cntPtr, val2 );
            Bin4_None<BlockSize>( cntPtr, val3 );
            Bin4_None<BlockSize>( cntPtr, val4 );


            //-----
            // Read & Bin [13..16]
            //-----

            val1 = inPtr[(12u*nThreadsPerGrid)];
            val2 = inPtr[(13u*nThreadsPerGrid)];
            val3 = inPtr[(14u*nThreadsPerGrid)];
            val4 = inPtr[(15u*nThreadsPerGrid)];

            Bin4_None<BlockSize>( cntPtr, val1 );
            Bin4_None<BlockSize>( cntPtr, val2 );
            Bin4_None<BlockSize>( cntPtr, val3 );
            Bin4_None<BlockSize>( cntPtr, val4 );


            // Move to next section
            baseIdx        += (16u * nThreadsPerGrid);
            nLeftOverElems -= (16u * nThreadsPerGrid);
         }
      }


      //------
      // Try Section of 8
      //------

      if (K_length >= 8u)
      {
         // Process 8 chunks safely without range checking
         if (nLeftOverElems >= (8u * nThreadsPerGrid))
         {
            // Get pointer
            inPtr = &inVals[baseIdx];

            //-----
            // Read & Bin [1..4]
            //-----

            val1 = inPtr[(0u*nThreadsPerGrid)];
            val2 = inPtr[(1u*nThreadsPerGrid)];
            val3 = inPtr[(2u*nThreadsPerGrid)];
            val4 = inPtr[(3u*nThreadsPerGrid)];

            Bin4_None<BlockSize>( cntPtr, val1 );
            Bin4_None<BlockSize>( cntPtr, val2 );
            Bin4_None<BlockSize>( cntPtr, val3 );
            Bin4_None<BlockSize>( cntPtr, val4 );


            //-----
            // Read & Bin [5..8]
            //-----

            val1 = inPtr[(4u*nThreadsPerGrid)];
            val2 = inPtr[(5u*nThreadsPerGrid)];
            val3 = inPtr[(6u*nThreadsPerGrid)];
            val4 = inPtr[(7u*nThreadsPerGrid)];

            Bin4_None<BlockSize>( cntPtr, val1 );
            Bin4_None<BlockSize>( cntPtr, val2 );
            Bin4_None<BlockSize>( cntPtr, val3 );
            Bin4_None<BlockSize>( cntPtr, val4 );


            // Move to next section
            baseIdx        += (8u * nThreadsPerGrid);
            nLeftOverElems -= (8u * nThreadsPerGrid);
         }
      }


      //------
      // Try Section of 4
      //------

      if (K_length >= 4u)
      {
         // Process 4 chunks safely without range checking
         if (nLeftOverElems >= (4u * nThreadsPerGrid))
         {
            // Get pointer
            inPtr = &inVals[baseIdx];

            //-----
            // Read & Bin [1..4]
            //-----

            val1 = inPtr[(0u*nThreadsPerGrid)];
            val2 = inPtr[(1u*nThreadsPerGrid)];
            val3 = inPtr[(2u*nThreadsPerGrid)];
            val4 = inPtr[(3u*nThreadsPerGrid)];

            Bin4_None<BlockSize>( cntPtr, val1 );
            Bin4_None<BlockSize>( cntPtr, val2 );
            Bin4_None<BlockSize>( cntPtr, val3 );
            Bin4_None<BlockSize>( cntPtr, val4 );


            // Move to next section
            baseIdx        += (4u * nThreadsPerGrid);
            nLeftOverElems -= (4u * nThreadsPerGrid);
         }
      }


      //------
      // Try Section of 2
      //------

      if (K_length >= 2u)
      {
         // Process 2 chunks safely without range checking
         if (nLeftOverElems >= (2u * nThreadsPerGrid))
         {
            // Get pointer
            inPtr = &inVals[baseIdx];

            //-----
            // Read & Bin [1..2]
            //-----

            val1 = inPtr[(0u*nThreadsPerGrid)];
            val2 = inPtr[(1u*nThreadsPerGrid)];

            Bin4_None<BlockSize>( cntPtr, val1 );
            Bin4_None<BlockSize>( cntPtr, val2 );


            // Move to next section
            baseIdx        += (2u * nThreadsPerGrid);
            nLeftOverElems -= (2u * nThreadsPerGrid);
         }
      }


      //------
      // Try Section of 1
      //------

      if (K_length >= 1u)
      {
         // Process 1 chunk safely without range checking
         if (nLeftOverElems >= (1u * nThreadsPerGrid))
         {
            // Get pointer
            inPtr = &inVals[baseIdx];

            //-----
            // Read & Bin [1]
            //-----

            val1 = inPtr[(0u*nThreadsPerGrid)];
            Bin4_None<BlockSize>( cntPtr, val1 );

            // Move to next section
            baseIdx        += (1u * nThreadsPerGrid);
            nLeftOverElems -= (1u * nThreadsPerGrid);
         }
      }


      //------
      // Process Last few elements
      //    while carefully RANGE CHECKING !!!
      //------

      if (nLeftOverElems > 0u)
      {
         // Make sure we are 'in range' before reading
         uint inRange1 = (baseIdx <= stop);

         // Read in 32-bit element, only if *safely* in range
         if (inRange1) { val1 = inVals[baseIdx]; }

         // Bin 'four' values in 32-bit element (4 bytes at a time)
         if (inRange1) { Bin4_None<BlockSize>( cntPtr, val1 ); }
      }

      // Update Accumulation count
      overflow += K4_length;   // 64 = 16 elems * 4 bytes per elem
	}


   //-----
   // Accumulate 'thread' counts into 'row' counts
   //    Note: Also zeros out 'per thread' count array
   //-----

   if (overflow > 0u)
   {
      // Sync Threads in Block
      if (WarpsPerBlock >= 2u) { __syncthreads(); }

      //AddThreadToRowCounts_V1< BlockSize, BlockMask >( rowCnt1, rowCnt2, rowCnt3, rowCnt4, basePtr, tid );
      AddThreadToRowCounts_V2< BlockSize, BlockMask >( rowCnt1, rowCnt2, rowCnt3, rowCnt4, basePtr, tid );
      overflow = 0u;

      // Sync Threads in Block
      if (WarpsPerBlock >= 2u) { __syncthreads(); }
   }


	//-------------------------------------------------
	// Write out final row 'counts'
	//-------------------------------------------------

   {
      // Compute starting 'row counts' offset
      uint rIdx = threadIdx.x * 4u;         // 4 groups per lane
      uint rRow = rIdx >> logBankSize;
      uint rCol = rIdx & BankMask;

      uint rowIdx = (rRow * strideBank) + (rCol + 1u);
         // Extra '+1' to shift past initial pad element      

      uint * rowPtr = &s_thrdCounts[rowIdx];

      // Store row counts in row array
      rowPtr[0] = rowCnt1;
      rowPtr[1] = rowCnt2;
      rowPtr[2] = rowCnt3;
      rowPtr[3] = rowCnt4;

      // Sync Threads in Block
      if (WarpsPerBlock >= 2u) { __syncthreads(); }

      // Get Warp Row & Column
      uint warpRow = threadIdx.x >> logWarpSize; // tid / 32
      uint warpCol = threadIdx.x & WarpMask;     // tid % 32

      // Get local & global indices
      uint outGlobal = (blockIdx.x * nHistBins);
      uint outLocal  = (warpRow * OutWarpSize);
      uint rowBase   = (warpRow * OutStrideSize);
      uint outBase   = outGlobal + outLocal;
      uint rowOff    = warpCol + 1u;

      uint outIdx = outBase + warpCol;
          rowIdx = rowBase + rowOff;

      // Get local & global pointers
      uint * outPtr = &outRowCounts[outIdx];
            rowPtr = &s_thrdCounts[rowIdx];

         // Write our 'per row' counts in warp sequential order
      if (OutLength >= 1u) { outPtr[(0u*WarpSize)] = rowPtr[(0u*strideBank)]; }
      if (OutLength >= 2u) { outPtr[(1u*WarpSize)] = rowPtr[(1u*strideBank)]; }
      if (OutLength >= 3u) { outPtr[(2u*WarpSize)] = rowPtr[(2u*strideBank)]; }
      if (OutLength >= 4u) { outPtr[(3u*WarpSize)] = rowPtr[(3u*strideBank)]; }
      if (OutLength >= 5u) { outPtr[(4u*WarpSize)] = rowPtr[(4u*strideBank)]; }
      if (OutLength >= 6u) { outPtr[(5u*WarpSize)] = rowPtr[(5u*strideBank)]; }
      if (OutLength >= 7u) { outPtr[(6u*WarpSize)] = rowPtr[(6u*strideBank)]; }
      if (OutLength >= 8u) { outPtr[(7u*WarpSize)] = rowPtr[(7u*strideBank)]; }
   }
}


//-----------------------------------------------
// Name: H_K2_RowCounts_To_RowStarts_256
// Desc: Sum 256-way 'per row' counts into 
//       total 256-way counts using prefix-sum
//------------------------------------------------

template < uint logBankSize,		// log<2>( Channels per Bank )
           uint logWarpSize,		// log<2>( Threads Per Warp )
           uint BlockSize >	      // Threads Per Block
__global__
void H_K2_RowCounts_To_RowStarts_256
( 
         uint * outTotalCounts,	// OUT - total counts
         uint * outTotalStarts,	// OUT - total starts
         uint * outRowStarts,	   // OUT - row starts
	const uint * inRowCounts,	   // IN  - 'per row' counts to accumulate
         uint   nRows			      // IN  - number of rows to accumulate
) 
{
	//------------------------------------
	// Constant values
	//------------------------------------

		// Memory Channels Per Bank
	const uint BankSize  = 1u << logBankSize;	// 32 (or 16)
	const uint BankMask  = BankSize - 1u;	   // 31 (or 15)

		// Threads Per Warp
	const uint WarpSize  = 1u << logWarpSize;	// 32
	const uint WarpMask  = WarpSize - 1u;      // 31

		// Warps Per Block
	const uint WarpsPerBlock = BlockSize / WarpSize; // 8 = 256 / 32
	
		// Size of 'Row Counts' and 'Row Starts' array
	//const uint nElemsCounts = 256;
	//const uint banksCounts  = (nElemsCounts + BankMask) / BankSize;
	//const uint padCounts    = ((banksCounts * BankSize) - nElemsCounts);
	//const uint sizeCounts   = nElemsCounts + padCounts;

      // Stride for padded bank of elements
   const uint strideBank = 1u + BankSize;

		// Serial Scan Array
   const uint nSS1      = 256u + 2u;
   const uint nRowsSS1  = (nSS1 + BankMask) / BankSize;
	const uint nElemsSS1 = nRowsSS1 * strideBank;
	const uint banksSS1  = (nElemsSS1 + BankMask) / BankSize;
	const uint padSS1    = ((banksSS1 * BankSize) - nElemsSS1);
	const uint sizeSS1   = nElemsSS1 + padSS1;

		// WarpScan array
	const uint strideWS2 = WarpSize
		                   + (WarpSize >> 1u)
						       + 1u;			// 49 = (32 + 16 + 1)
   const uint nWarpsWS2 = 1u;
	const uint nElemsWS2 = nWarpsWS2 * strideWS2;
	const uint banksWS2  = (nElemsWS2 + BankMask) / BankSize;
	const uint padWS2    = ((banksWS2 * BankSize) - nElemsWS2);
	const uint sizeWS2   = nElemsWS2 + padWS2;

	//const uint nSafePassesCnts = sizeCounts / BlockSize;
	//const uint leftOverCnts    = sizeCounts - (nSafePassesCnts * BlockSize);

	const uint nSafePassesSS1  = sizeSS1 / BlockSize;
	const uint leftOverSS1     = sizeSS1 - (nSafePassesSS1 * BlockSize);

	const uint nSafePassesWS2  = sizeWS2 / BlockSize;
	const uint leftOverWS2     = sizeWS2 - (nSafePassesWS2 * BlockSize);


	//------------------------------------
	// Local variables
	//------------------------------------

		// shared memory
	//__shared__ uint s_rowStarts[sizeCounts];	// 'Row Starts' one chunk at a time
   __shared__ uint s_ss1[sizeSS1];            // Used for serial scan
	__shared__ uint s_ws2[sizeWS2];		      // Used for parallel warp scan

		// Registers
	uint tSum;				// Per thread accumulator

	//------------------------------------
	// Compute Indices & Pointers
	//------------------------------------

   uint warpRow, warpCol;
   uint storeIdx, prevIdx, ss1Idx, ws2Idx;
   {
      // Compute Bank Offsets
	   //uint bankRow = threadIdx.x >> logBankSize;		// tid / 32
	   uint bankCol = threadIdx.x & BankMask;			// tid % 32

	   // Compute warp offsets
	   warpRow = threadIdx.x >> logWarpSize;		// tid / 32
	   warpCol = threadIdx.x & WarpMask;			// tid % 32

      // Compute Store index (for storing final counts before prefix sum)
      uint sIdx = threadIdx.x;
      uint storeRow = sIdx >> logBankSize;   // tid / 32
      uint storeCol = sIdx & BankMask;       // tid % 32
      storeIdx = (storeRow * strideBank)
                 + storeCol
                 + 2u;        // Pad for 'reach back'

	      //--
	      // Previous Column (Serial Scan 1)
	      //   1.) Reach back one column
	      //   2.) But, we need to skip over extra padding before the first
         //       thread in every bank, so reach back two columns
         // However, the very first thread in the very first bank needs
         // to be able to reach back safely 2 columns without going 'out of range'.
         //
         // We work around this by pre-padding the 's_ss1' array with
         // an extra 2 elements and shifting indices over by two as needed to skip over padding.
	      //--

 	   uint prevCol = ((bankCol == 0u) ? 2u : 1u);
      prevIdx = storeIdx - prevCol;

      // Compute Serial Scan index
      uint ssIdx  = threadIdx.x * 8u;
      uint ss1Row = ssIdx >> logBankSize;   // (tid*8) / 32
      uint ss1Col = ssIdx & BankMask;       // (tid*8) % 32
      ss1Idx = (ss1Row * strideBank)
               + ss1Col
               + 2u;       // pad for 'reach back'

	   // Compute Warp Scan Index
	   ws2Idx  = (warpRow * strideWS2) 
		          + (WarpSize >> 1u)
		          + warpCol;
	}


	//------------------------------------
	// Zero out 'arrays'
	//------------------------------------

   uint * setPtr = NULL;

	//-
	// Zero out 'row starts' array
	//-

   //setPtr = (&s_rowStarts[0]);
   //SetArray_BlockSeq
   //   < 
   //      uint, BlockSize, nSafePassesCnts, 
   //      leftOverCnts, sizeCounts 
   //   >
   //   ( 
   //      setPtr, 0u
   //   );


   //-
	// Zero out 'Serial Scan' array
	//-

   setPtr = (&s_ss1[0]);
   SetArray_BlockSeq
      < 
         uint, BlockSize, nSafePassesSS1, 
         leftOverSS1, sizeSS1 
      >
      ( 
         setPtr, 0u
      );


   //-
	// Zero out 'Warp Scan' array
	//-

   setPtr = (&s_ws2[0]);
   SetArray_BlockSeq
      < 
         uint, BlockSize, nSafePassesWS2, 
         leftOverWS2, sizeWS2 
      >
      ( 
         setPtr, 0u
      );


   // Sync all threads in block
   if (WarpsPerBlock > 2u) { __syncthreads(); }


	//-------------------------------------------------
   // Phase 1:
	//   Serial Reduction of all rows of 'per row' counts
	//	  down to single set of 'total' counts
	//-------------------------------------------------

   {
      const uint * inPtr = &inRowCounts[threadIdx.x];

	   // Initialize 'Thread Sum' to identity value
	   tSum = 0;

	   // Loop over row counts
	   #pragma unroll
	   for (uint currPass = 0u; currPass < nRows; currPass++)
	   {		
		   // Grab count from global arrary
		   uint currCnt = inPtr[0];

		   // Accumulate 'per row' counts into a 'total' count
		   tSum = tSum + currCnt;

		   // Move to next set of 'row counts' to process
         inPtr += BlockSize;
	   }

	   // Store the 'total count's
	   outTotalCounts[threadIdx.x] = tSum;

	   // Also store 'total count's into 'Serial Scan' array
      s_ss1[storeIdx] = tSum;

      // Sync all threads in block
      if (WarpsPerBlock > 2u) { __syncthreads(); }
   }


	//--------------------------------------
   // Phase 2:
	//   convert 'total counts' into 'total starts'
   //   using prefix sum
   //--------------------------------------

   if (warpRow == 0)
   {
	   volatile uint * wsPtr = (uint *)&(s_ws2[0]);
   	
      uint * SS1_ptr = &s_ss1[ss1Idx];

		   // For higher performance, we use registers instead of shared memory
		   // Tradeoff - lots of register pressure (8 registers per thread)
      uint ss01, ss02, ss03, ss04;
      uint ss05, ss06, ss07, ss08;

      //-----
      // Serial Scan (on short sequence of 8 values)
      //-----

      // Grab short sequence of 8 values from ss1 array
      ss01 = SS1_ptr[0];
      ss02 = SS1_ptr[1];
      ss03 = SS1_ptr[2];
      ss04 = SS1_ptr[3];
      ss05 = SS1_ptr[4];
      ss06 = SS1_ptr[5];
      ss07 = SS1_ptr[6];
      ss08 = SS1_ptr[7];

      // Serial scan short sequence (in registers)
      //ss01 = <identity> + ss01;
      ss02 = ss01 + ss02;
      ss03 = ss02 + ss03;
      ss04 = ss03 + ss04;
      ss05 = ss04 + ss05;
      ss06 = ss05 + ss06;
      ss07 = ss06 + ss07;
      ss08 = ss07 + ss08;

      //-
      // Store final serial scan result into warp scan array
      //-

      uint wi = ws2Idx;
      tSum = ss08;
      wsPtr[wi] = tSum;

	   //-----
	   // Warp Scan (on 32 threads in parallel)
	   //-----

      wsPtr[wi] = tSum = wsPtr[wi -  1u] + tSum;
      wsPtr[wi] = tSum = wsPtr[wi -  2u] + tSum;
      wsPtr[wi] = tSum = wsPtr[wi -  4u] + tSum;
      wsPtr[wi] = tSum = wsPtr[wi -  8u] + tSum;
      wsPtr[wi] = tSum = wsPtr[wi - 16u] + tSum;


      //-----
      // Serial Update (on short sequence of 8 values)
      //-----

      //-
      // Grab update (prefix) value from Warp Array
      //-
         // Note:  Need to reach back 'one column' to get exclusive result
      uint prevWI = wi - 1u;
      tSum = wsPtr[prevWI];


      //-
      // Update each element short sequence with prefix (in registers)
      //-

      ss01 = tSum + ss01;
      ss02 = tSum + ss02;
      ss03 = tSum + ss03;
      ss04 = tSum + ss04;
      ss05 = tSum + ss05;
      ss06 = tSum + ss06;
      ss07 = tSum + ss07;
      ss08 = tSum + ss08;

      // Store 'prefix sum' results back in 'serial scan' array
      SS1_ptr[0] = ss01;
      SS1_ptr[1] = ss02;
      SS1_ptr[2] = ss03;
      SS1_ptr[3] = ss04;
      SS1_ptr[4] = ss05;
      SS1_ptr[5] = ss06;
      SS1_ptr[6] = ss07;
      SS1_ptr[7] = ss08;
   } // end warpRow == 0

   // Sync all threads in block
   if (WarpsPerBlock > 2u) { __syncthreads(); }


   //-----
   // Grab starting 'row start' (total sum) for this thread
   //    Note #1:  Need to 'reach back' one column for exclusive results
   //    Note #2:  This will result in an unavoidable '2-way' bank conflict
   //-----

   uint rowSum = s_ss1[prevIdx];

	// Store total starts (from previous column)
	outTotalStarts[threadIdx.x] = rowSum;

   // Sync all threads in block
   if (WarpsPerBlock > 2u) { __syncthreads(); }


	//-------------------------------------------------
   // Phase 3:
   //    Accumulate and write out 'per row' starts
	//-------------------------------------------------

   {
      const uint * inPtr  = &inRowCounts[threadIdx.x];
            uint * outPtr = &outRowStarts[threadIdx.x];

	   // Initialize 'Thread Sum' to identity value

	   // Loop over row counts
	   #pragma unroll
	   for (uint currPass = 0u; currPass < nRows; currPass++)
	   {		
		   // Read 'in' current count from global arrary
		   uint currCnt = inPtr[0];

         // Write 'out' current row sum to global array
         outPtr[0] = rowSum;

		   // Accumulate 'per row' count into running 'row sum' start
		   rowSum = rowSum + currCnt;

         //-
		   // Move to next row
         //-
         
         inPtr  += BlockSize;
         outPtr += BlockSize;
	   }
      // Sync all threads in block
      //if (WarpsPerBlock > 2u) { __syncthreads(); }
   }
}


////////////////////////////////////////////////////////////////////////////////
// Host interface to GPU TRISH histogram
////////////////////////////////////////////////////////////////////////////////


/*-----------------
  Local Defines
-----------------*/

// GTX 560M
//#define NUM_GPU_SMs (4u)

// TESLA 2050 (2070)
//#define NUM_GPU_SMs (14u)

// GTX 480
#define NUM_GPU_SMs (15u)

// GTX 580
//#define NUM_GPU_SMs (16u)


// Intermediate CUDA buffers
static uint * d_rowCounts = NULL;
static uint * d_rowStarts = NULL;
static uint * d_totalStarts = NULL;


//-----------------------------------------------
// Name:  initTrish256
// Desc:  Initialize intermediate GPU Buffers
//-----------------------------------------------

extern "C" 
void initTrish256(void)
{
	//------
	// Local Constants
	//------

	const uint nHistBins256  = 256u;
	const uint nGPU_SMs      = NUM_GPU_SMs;
	const uint nGPU_ConcurrentBlocks = 3u;
	const uint K1_GridSize   = nGPU_SMs * nGPU_ConcurrentBlocks;
	const uint K1_nRows      = K1_GridSize;
	const uint sizeRowCounts = K1_nRows * nHistBins256 * sizeof(uint);
	const uint sizeTotal     = nHistBins256 * sizeof(uint);

	// Create intermediate GPU buffers
    cutilSafeCall( cudaMalloc( (void **)&d_rowCounts, sizeRowCounts ) );
    cutilSafeCall( cudaMalloc( (void **)&d_rowStarts, sizeRowCounts ) );
    cutilSafeCall( cudaMalloc( (void **)&d_totalStarts, sizeTotal ) );
}



//-----------------------------------------------
// Name:  closeTrish256
// Desc:  cleanup intermediate GPU buffers
//-----------------------------------------------

extern "C" 
void closeTrish256(void)
{
	// Destroy Intermediate GPU buffers
    cutilSafeCall( cudaFree( d_totalStarts ) );
	cutilSafeCall( cudaFree( d_rowStarts ) );
	cutilSafeCall( cudaFree( d_rowCounts ) );
}



//---------------------------------------------------------
// Name:  histogramTrish256
// Desc:  CPU Wrapper function around GPU kernels 
//        for use in "histogram" demo  
//---------------------------------------------------------

extern "C" 
void histogramTrish256
(
	// Function Parameters
    uint *d_Histogram,	// OUT - Final 256-way histogram counts
    void *d_Data,		//  IN - input data to bin & count into histogram
    uint byteCount		//  In - length of input data array
)
{
	//-----
	// Local Constants=
	//-----


      // Note:  The best # of blocks for the TRISH algorithm appears to be
      //        The # of SM's on the card * the number of concurrent blocks.
      //        This is the mininum to effectively use all hardware resources effectively.
      // 
      // For Example:  On the following Fermi cards, the grid sizes for best performance would be ... 
      //  GTX 560M    = 12 =  4 * 3
      //  TELSA M2050 = 42 = 14 * 3
      //  GTX 480     = 45 = 15 * 3
      //  GTX 580     = 48 = 16 * 3

	const uint nGPU_SMs     = NUM_GPU_SMs;	// See #defines above
	const uint nGPU_ConcurrentBlocks = 3u;	// for Fermi architectures, we can achieve 3 concurrent blocks per SM (64 * 3 = 192 => 192/1536 => 12.5% occupancy 
	const uint logBankSize  = 5u;		//  5 = log<2>( Memory Banks )
	const uint logWarpSize  = 5u;       //  5 = log<2>( Threads per Warp )
	
	const uint K1_BlockSize = 64u;      // 64 = Threads per Block (Histogram Kernel)
	const uint K1_GridSize  = nGPU_SMs * nGPU_ConcurrentBlocks;	 // GridSize (Histogram Kernel)

	const uint K2_BlockSize = 256u;		// 256 = Threads per Block (RowSum Kernel)
	const uint K2_GridSize  = 1u;		//  1 = GridSize (RowSum Kernel)
	
	const uint K1_Length    = 31u;		//  31 = Work Per thread (loop unrolling)
	const uint in_start     = 0u;		//   0 = starting range
	const uint K1_nRows     = K1_GridSize;	//  ?? = Number of rows (blocks) that are cooperatively striding across input data set


	//-----
	// Get number of elements
	//-----

    assert( byteCount > 0u );
    assert( byteCount % sizeof(uint) == 0u );

	uint nElems = byteCount >> 2u;  // byteCount/4
	uint in_stop = nElems - 1u;	

	const uint * d_inVals = (const uint *)d_Data;


	/*--------------------------------------
	  Step 0. Create Intermediate buffers 
    --------------------------------------*/

	// Code moved to initTrish256() above


	/*------------------------------------------------------
	  Step 1. Bin & count elements into 'per row' 256-way histograms
	------------------------------------------------------*/

	H_K1_CountRows_256_TRISH
		< 
		  // Template Parameters
		  logBankSize,		// log<2>( Memory Banks ) 
		  logWarpSize,		// log<2>( Threads per Warp )
		  K1_BlockSize,		// Threads per Block
		  K1_GridSize,      // Blocks per Grid
		  K1_Length			// Work Per Thread (Loop unrolling)
		>
		<<< 
			// CUDA CTA Parameters
			K1_GridSize,	// Blocks per Grid 
			K1_BlockSize	// Threads per Block
		>>>
		(
			// Function parameters
			d_rowCounts,	// IN - 'per row' histograms
			d_inVals,		// IN - 'input' data to count & bin
			in_start,		// IN - input range [start, stop] 
			in_stop			//      ditto
		);
   // Check if kernel execution generated an error    
   cutilCheckMsg( "H_K1_CountRows_256() Kernel execution failed!" );


	/*------------------------------------------------------
	    Step 2. Sum 'per row' histograms into 'final' 256-bin histogram
	------------------------------------------------------*/

    H_K2_RowCounts_To_RowStarts_256
		< 
			// Template Parameters
			logBankSize,	// log<2>( Memory Banks ) 
			logWarpSize,	// log<2>( Warp Size )
			K2_BlockSize	// Threads per Block
		>
        <<< 
			// CUDA CTA Parameters
			K2_GridSize,	// Blocks per Grid 
			K2_BlockSize	// Threads per Block
		>>>	
        (
			// Function parameters
			d_Histogram,    // OUT - Histogram Counts
			d_totalStarts,  // OUT - Histogram Starts
			d_rowStarts,    // OUT - 'Per Row' Histogram Starts
			d_rowCounts,    // IN  - 'Per Row' Histogram Counts
			K1_nRows		// IN  - number of rows
        );
	// Check if kernel execution generated an error    
	cutilCheckMsg( "H_K2_RowCounts_To_RowStarts_256() Kernel execution failed!" );


	/*--------------------------------------
	  Step 3. Cleanup intermediate buffers
	--------------------------------------*/

	// Code moved to closeTrish256() above
}